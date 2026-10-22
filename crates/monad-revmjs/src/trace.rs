use monad_revm::{extension, MonadHardfork};
use revm::{
    bytecode::opcode::OpCode,
    interpreter::{
        interpreter_types::{InputsTr, Jumps, LegacyBytecode, MemoryTr},
        Interpreter, InterpreterAction,
    },
    primitives::{hex, keccak256},
    Inspector,
};
use serde::Serialize;
use std::collections::BTreeMap;

const MAX_STEPS: usize = 100_000;
const MAX_TRACE_BYTES: usize = 32 * 1024 * 1024;

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
    pub(super) truncated: bool,
    spec: MonadHardfork,
    bytes: usize,
    pending: Option<usize>,
}

impl Trace {
    pub(super) const fn new(spec: MonadHardfork) -> Self {
        Self {
            steps: Vec::new(),
            preimages: BTreeMap::new(),
            truncated: false,
            spec,
            bytes: 0,
            pending: None,
        }
    }
}

impl<CTX> Inspector<CTX> for Trace {
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
        if opcode == 0x20 && interp.stack.data().len() >= 2 {
            let stack = interp.stack.data();
            let offset = usize::try_from(stack[stack.len() - 1]);
            let len = usize::try_from(stack[stack.len() - 2]);
            if let (Ok(offset), Ok(len)) = (offset, len) {
                if let Some(end) = offset.checked_add(len) {
                    if end <= memory.len() {
                        let input = &memory[offset..end];
                        self.preimages.insert(hex::encode(keccak256(input)), hex::encode(input));
                    }
                }
            }
        }
        self.pending = Some(self.steps.len());
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
        if let Some(index) = self.pending.take() {
            let step = &mut self.steps[index];
            step.gas_cost = step.gas.saturating_sub(interp.gas.remaining());
            if let Some(InterpreterAction::Return(result)) = &interp.bytecode.action {
                if !result.result.is_ok() {
                    step.error = Some(format!("{:?}", result.result));
                }
            }
        }
    }
}
