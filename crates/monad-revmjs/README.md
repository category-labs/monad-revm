# monad-revmjs

An experimental JavaScript package for running `monad-revm` through WebAssembly. It provides a local in-memory VM, execution with optional commit, account reads, state snapshots and bounded instruction traces. The engine and opcode implementations come from the sibling `monad-revm` crate.

The package contains prebuilt WASM, an ES module loader and TypeScript declarations. Consumers need no Rust installation, native Node addon, WASI shim or package install scripts. Browser applications can load it inside a Web Worker. Node applications supply the WASM bytes explicitly.

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

## Building from this repository

Use the repository's `main` branch as the engine baseline. The wrapper is a separate workspace member; changing browser bindings does not require changing engine rules. The workspace lockfile pins build dependencies, and a dedicated `wasm-release` profile leaves native release settings unchanged.

Prerequisites: Node.js 22 or newer, Rust 1.91, its browser target, and the matching binding generator:

```sh
rustup toolchain install 1.91
rustup target add wasm32-unknown-unknown --toolchain 1.91
cargo +1.91 install wasm-bindgen-cli --version 0.2.128 --locked
```

From the repository root:

```sh
cargo +1.91 test -p monad-revmjs --locked
node crates/monad-revmjs/scripts/build.mjs
node crates/monad-revmjs/scripts/test-package.mjs
```

The build writes `target/monad-revmjs/package`. The package test packs that directory, installs the tarball into an isolated project with scripts disabled, then exercises the installed module. The resulting archive is `target/monad-revmjs/monad-revmjs-0.1.0-preview.1.tgz`. All generated artifacts are ignored by Git in the source branch. `build-info.json` records the source revision, dirty-state flag, toolchain and lockfile checksum.

On macOS, a compatible Xcode Command Line Tools SDK must be available; select one through `SDKROOT` if needed.

## Distribution

The initial Remix integration consumes a Git commit on the `monad-revmjs-preview` distribution branch. That branch contains only the generated npm package, with no Rust build step. Its `build-info.json` identifies the source commit on the `monad-revmjs` implementation branch based on `main`. Consumers pin the complete distribution commit SHA; do not depend on the moving branch name. Source changes are reviewed in a PR to `main`.

After package registry access is configured, the same tested package directory can be published with `npm publish target/monad-revmjs/package --tag next`. Validate a clean source revision, matching Cargo/npm versions, native tests and the packed-package tests before publishing. Then update consumers to an exact registry version. No automatic registry publication is configured by this change.
