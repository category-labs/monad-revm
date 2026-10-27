# JavaScript API

[Package overview](../README.md)

## Using the package

Initialize once before constructing an engine. In a browser or module worker:

```js
import init, { Engine } from 'monad-revmjs'

await init()
const engine = new Engine('monad-next', 1337)
engine.fund('0x1111111111111111111111111111111111111111', '0x3635c9adc5dea00000')
// Read state, execute requests and export snapshots through engine methods.
engine.free()
```

The loader locates `dist/monad_revmjs_bg.wasm` relative to its own module URL. Bundlers must preserve or emit that asset. An explicit URL or byte buffer can instead be passed as `await init({ module_or_path: ... })`.

In Node.js:

```js
import { readFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import init, { Engine } from 'monad-revmjs'

const require = createRequire(import.meta.url)
await init({ module_or_path: await readFile(require.resolve('monad-revmjs/wasm')) })
const engine = new Engine('monad-ten', 1337)
engine.free()
```

`Monad Ten` and experimental `Monad Next` are explicit choices. `Monad Next` enables the extension instructions supported by the engine revision; it is not a declaration of public-network activation.

## API and ownership

- `new Engine(spec, chainId)`: create an empty VM. This API currently accepts a 32-bit unsigned chain ID.
- `fund(address, balance)`: set a development balance, preserving code and nonce.
- `account(address)`: JSON containing balance, nonce, code and nonzero storage slots.
- `execute(requestJson)`: execute a JSON `ExecuteRequest`, returning JSON `ExecutionResult`. TypeScript interfaces for both are exported by the package.
- `set_block_hash(number, hash)`: supply a hash for `BLOCKHASH`; the current API accepts a 32-bit block number here. Unsupplied hashes return zero, including after snapshot restoration. The usual `BLOCKHASH` range restrictions still apply.
- `snapshot()` / `restore(snapshotJson)`: versioned engine state. Restore validates all fields and bytecode before replacing state; invalid snapshots throw a recoverable error and leave the engine unchanged.
- `free()`: release the engine's WASM allocation.

Execution requests explicitly supply the sender, transaction fields, block number, timestamp, block gas limit, base fee, beneficiary and prevrandao. The package has no funded accounts, wallet keys, RPC server, synthetic blockchain history or browser persistence policy. Those belong to the consuming application.

Transactions support types `0`, `1`, `2` and `4`. Set `type` explicitly when preserving a signed transaction's envelope; omitted types are inferred from authorization, priority-fee and access-list fields. `accessList` entries contain an address and 32-byte storage keys. `authorizationList` entries contain `chainId`, `address`, `nonce`, `yParity`, `r` and `s`; use exact hex quantities for authorization chain IDs, nonces and signature scalars. Signature recovery and authorization processing run in Rust. Invalid authorizations may be skipped according to EIP-7702; malformed transaction envelopes throw without committing state. Type 4 requires a destination, a priority fee and a nonempty authorization list. Applied delegations and authority nonces survive an execution revert.

For reserve-balance decisions, pass `chainContext.parentSendersAndAuthorities` and `grandparentSendersAndAuthorities`. Optional `currentBlockSenders` and `currentBlockAuthorities` describe preceding transactions in the current block and must have matching lengths. The engine appends this transaction's sender and recovered authorities. `authorizationAuthorities` in the result supplies those recoverable addresses for the consumer's block history, including tuples skipped for chain ID or nonce. Omitted history is empty and cannot reproduce historical reserve behavior.

```js
const result = JSON.parse(engine.execute(JSON.stringify({
  from: '0x1111111111111111111111111111111111111111',
  to: '0x2222222222222222222222222222222222222222',
  data: '0x', value: '0x0', gas: 100000, gasPrice: '0x1', nonce: 0,
  blockNumber: 1, timestamp: 1800000000, blockGasLimit: 30000000,
  baseFee: '0x0', beneficiary: '0x0000000000000000000000000000000000000000',
  prevrandao: '0x' + '00'.repeat(32), commit: false, trace: true
})))
```

Use exact hex strings for balances, transaction values, gas prices and hashes. JSON number fields must be safe integers. `commit: false` leaves engine state unchanged. Committed reverts still apply transaction-level effects such as nonce and gas charging. Validation errors throw; execution reverts and halts return a failed result.

`gasUsed` is the full submitted gas limit charged by Monad; `gasSpent` is actual consumption including intrinsic gas. Traces capture the state before each instruction, with real bytecode PCs, root depth 1, unprefixed hex stack words and contiguous hex memory. A two-byte extension occupies one trace step. `preimages` records inputs of successful `KECCAK256` instructions, including newly expanded zero-filled memory and empty inputs; keys and values are unprefixed hex. Traces are capped at 100,000 steps or approximately 32 MiB shared by steps and preimages; `traceTruncated` signals incomplete inspection without changing execution results. Consumers must not present a truncated trace as complete.

With `trace: true`, `consoleCalls` contains the raw `0x` calldata from `STATICCALL`s to the Hardhat console address, in execution order. Decode these payloads in the consumer using the console ABI. Console calls survive reverts and continue after instruction-trace truncation; they do not change execution results or gas accounting. With `trace: false`, `consoleCalls` is empty.

With tracing enabled, `codeByAddress` records the runtime bytecode actually executed at each called address, with EIP-7702 delegation resolved. It shares the trace byte budget and remains available for reverted or simulated executions. Debuggers can use this code after a later delegation change without querying current account code. Contract creation code remains in the creation frame's input. Without tracing, the map is empty.

The API is for a local development VM. It does not provide historical context discovery, staking system transitions, remote database callbacks or a JSON-RPC provider. Blob transactions are unsupported by Monad. Execution is synchronous; use a worker for browser responsiveness.
