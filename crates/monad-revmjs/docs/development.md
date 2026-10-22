# Building and distributing monad-revmjs

[Package overview](../README.md)

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
