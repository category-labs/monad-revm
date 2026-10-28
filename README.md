# monad-revmjs

An experimental JavaScript/WASM package for running `monad-revm` in browsers and
Node.js. It provides an in-memory VM, state snapshots and bounded instruction traces.
The package includes prebuilt WASM, an ES module loader and TypeScript declarations;
consumers need no Rust toolchain or package install scripts.

## Quick start

Use a [prebuilt preview package](docs/development.md#distribution) or
[build it from source](docs/development.md#building-from-this-repository).
In a browser or module worker:

```js
import init, { Engine } from 'monad-revmjs'

await init()
const engine = new Engine('monad-ten', 1337)
engine.fund('0x1111111111111111111111111111111111111111', '0x3635c9adc5dea00000')
console.log(JSON.parse(engine.account('0x1111111111111111111111111111111111111111')))
engine.free()
```

This is a local development VM: applications supply block context, persistence and
provider behavior. Execution is synchronous; use a worker in the browser. The optional
`monad-next` spec enables experimental extensions and does not imply network activation.

## Documentation

- [Browser/Node initialization, API, traces and limitations](docs/api.md)
- [Build, package tests and distribution](docs/development.md)
- [Local staking transitions and parity tests](docs/staking.md)
