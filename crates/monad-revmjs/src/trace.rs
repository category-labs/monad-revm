use monad_revm::{extension, MonadHardfork};
use revm::{
    bytecode::opcode::{OpCode, KECCAK256},
    context_interface::ContextTr,
    interpreter::{
        interpreter_types::{InputsTr, Jumps, LegacyBytecode, MemoryTr},
        CallInputs, CallOutcome, CallScheme, Interpreter, InterpreterAction,
    },
    primitives::{address, hex, Address, U256},
    Inspector,
};
use serde::Serialize;
use std::{collections::BTreeMap, ops::Range};

const MAX_STEPS: usize = 100_000;
const MAX_TRACE_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug)]
struct PendingStep {
    index: usize,
    hash_input: Option<Range<usize>>,
}

fn hash_input_range(stack: &[U256]) -> Option<Range<usize>> {
    let [.., len, offset] = stack else { return None };
    let len = usize::try_from(*len).ok()?;
    // A zero-length hash ignores the offset, even if it does not fit in usize.
    let start = if len == 0 { 0 } else { usize::try_from(*offset).ok()? };
    Some(start..start.checked_add(len)?)
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct Step {
    pc: usize,
    op: String,
    gas: u64,
    gas_cost: u64,
    depth: usize,
    stack: Vec<String>,
    memory: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug)]
pub(super) struct Trace {
    pub(super) steps: Vec<Step>,
    pub(super) preimages: BTreeMap<String, String>,
    pub(super) code_by_address: BTreeMap<String, String>,
    pub(super) console_calls: Vec<String>,
    pub(super) truncated: bool,
    spec: MonadHardfork,
    bytes: usize,
    pending: Option<PendingStep>,
    pending_code_address: Option<Address>,
}

impl Trace {
    pub(super) const fn new(spec: MonadHardfork) -> Self {
        Self {
            steps: Vec::new(),
            preimages: BTreeMap::new(),
            code_by_address: BTreeMap::new(),
            console_calls: Vec::new(),
            truncated: false,
            spec,
            bytes: 0,
            pending: None,
            pending_code_address: None,
        }
    }

    fn record_preimage(&mut self, interp: &Interpreter, input: Range<usize>) {
        let Some(hash) = interp.stack.data().last() else { return };
        let hash = format!("{hash:064x}");
        if self.preimages.contains_key(&hash) {
            return;
        }
        let size = 128 + input.len() * 2;
        if self.bytes + size > MAX_TRACE_BYTES {
            self.truncated = true;
            return;
        }
        self.bytes += size;
        self.preimages.insert(hash, hex::encode(&*interp.memory.slice(input)));
    }
}

impl<CTX: ContextTr> Inspector<CTX> for Trace {
    fn initialize_interp(&mut self, interp: &mut Interpreter, _ctx: &mut CTX) {
        // Capture the executed code, including resolved EIP-7702 delegation.
        // Creation code is already recoverable from the creation frame's input.
        let original_address = self.pending_code_address.take();
        let Some(resolved_address) = interp.input.bytecode_address() else { return };
        let address = original_address.unwrap_or(*resolved_address);
        let address = format!("{address:#x}");
        if self.truncated || self.code_by_address.contains_key(&address) {
            return;
        }
        let code = interp.bytecode.bytecode_slice();
        let size = 96 + code.len() * 2;
        if self.bytes + size > MAX_TRACE_BYTES {
            self.truncated = true;
            return;
        }
        self.bytes += size;
        self.code_by_address.insert(address, hex::encode_prefixed(code));
    }

    fn call(&mut self, context: &mut CTX, inputs: &mut CallInputs) -> Option<CallOutcome> {
        // Monad resolves inputs.bytecode_address to the delegation target after this hook.
        self.pending_code_address = Some(inputs.bytecode_address);
        if inputs.scheme == CallScheme::StaticCall
            && inputs.target_address == address!("000000000000000000636f6e736f6c652e6c6f67")
        {
            // Console output is diagnostic data and survives reverts and trace truncation.
            self.console_calls.push(hex::encode_prefixed(&*inputs.input.as_bytes(context)));
        }
        None
    }

    fn step(&mut self, interp: &mut Interpreter, _ctx: &mut CTX) {
        self.pending = None;
        let size = 128 + interp.stack.data().len() * 66 + interp.memory.size() * 2;
        if self.truncated || self.steps.len() >= MAX_STEPS || self.bytes + size > MAX_TRACE_BYTES {
            self.truncated = true;
            return;
        }
        self.bytes += size;
        let pc = interp.bytecode.pc();
        let opcode = interp.bytecode.opcode();
        let op = if opcode == extension::EXTENSION && self.spec == MonadHardfork::MonadNext {
            match interp.bytecode.bytecode_slice().get(pc + 1) {
                Some(&extension::CALLSTACKDEPTH_SELECTOR) => "CALLSTACKDEPTH",
                Some(&extension::CALLERN_SELECTOR) => "CALLERN",
                _ => "INVALID",
            }
        } else {
            OpCode::new(opcode).map_or("INVALID", OpCode::as_str)
        };
        let memory = interp.memory.slice(0..interp.memory.size());
        self.pending = Some(PendingStep {
            index: self.steps.len(),
            hash_input: if opcode == KECCAK256 {
                hash_input_range(interp.stack.data())
            } else {
                None
            },
        });
        self.steps.push(Step {
            pc,
            op: op.to_owned(),
            gas: interp.gas.remaining(),
            gas_cost: 0,
            depth: interp.input.depth() + 1,
            stack: interp.stack.data().iter().map(|word| format!("{word:064x}")).collect(),
            memory: hex::encode(&*memory),
            error: None,
        });
    }

    fn step_end(&mut self, interp: &mut Interpreter, _ctx: &mut CTX) {
        if let Some(pending) = self.pending.take() {
            let step = &mut self.steps[pending.index];
            step.gas_cost = step.gas.saturating_sub(interp.gas.remaining());
            if let Some(InterpreterAction::Return(result)) = &interp.bytecode.action {
                if !result.result.is_ok() {
                    step.error = Some(format!("{:?}", result.result));
                }
            }
            if interp.bytecode.action.is_none() {
                if let Some(input) = pending.hash_input {
                    self.record_preimage(interp, input);
                }
            }
        }
    }
}
