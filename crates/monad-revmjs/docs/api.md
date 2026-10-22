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
- `set_block_hash(number, hash)`: supply a hash for `BLOCKHASH`; the current API accepts a 32-bit block number here.
- `snapshot()` / `restore(snapshotJson)`: versioned engine state. Restore validates before replacing state.
- `free()`: release the engine's WASM allocation.

Execution requests explicitly supply the sender, transaction fields, block number, timestamp, block gas limit, base fee, beneficiary and prevrandao. The package has no funded accounts, wallet keys, RPC server, synthetic blockchain history or browser persistence policy. Those belong to the consuming application.

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

`gasUsed` is the full submitted gas limit charged by Monad; `gasSpent` is actual consumption including intrinsic gas. Traces capture the state before each instruction, with real bytecode PCs, root depth 1, unprefixed hex stack words and contiguous hex memory. A two-byte extension occupies one trace step. Traces are capped at 100,000 steps or approximately 32 MiB; `traceTruncated` signals incomplete inspection without changing execution results. Consumers must not present a truncated trace as complete.

The API is for a local development VM. It does not yet expose canonical historical chain context, reserve-balance metadata, staking system transitions, remote database callbacks, access lists, authorization lists or blob transactions. It does not provide a JSON-RPC provider. Execution is synchronous; use a worker for browser responsiveness.
