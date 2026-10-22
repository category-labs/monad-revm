import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { test } from 'node:test'
import { pathToFileURL } from 'node:url'

const require = createRequire(path.join(process.env.MONAD_REVMJS_TEST_PROJECT, 'package.json'))
const { default: init, Engine } = await import(pathToFileURL(require.resolve('monad-revmjs')))
await init({ module_or_path: readFileSync(require.resolve('monad-revmjs/wasm')) })
const sender = '0x1111111111111111111111111111111111111111'
const zero = '0x' + '00'.repeat(32)
const emptyAddress = '0x' + '00'.repeat(20)

function engine(spec = 'monad-next') {
  const vm = new Engine(spec, 1337)
  vm.fund(sender, '0x3635c9adc5dea00000')
  return vm
}
function execute(vm, to, data = '0x', options = {}) {
  return JSON.parse(vm.execute(JSON.stringify({
    from: sender, to, data, value: '0x0', gas: 1_000_000, gasPrice: '0x1',
    nonce: JSON.parse(vm.account(sender)).nonce, blockNumber: 1, timestamp: 1_800_000_000,
    blockGasLimit: 30_000_000, baseFee: '0x0', beneficiary: emptyAddress,
    prevrandao: zero, commit: false, trace: true, ...options
  })))
}
function deploy(vm, runtime) {
  const len = (runtime.length / 2).toString(16).padStart(2, '0')
  const result = execute(vm, null, `0x60${len}600c60003960${len}6000f3${runtime}`, { commit: true })
  assert.equal(result.success, true)
  return result.createdAddress
}

test('packed module executes, simulates, commits and restores state', () => {
  const vm = engine()
  const restored = engine()
  try {
    const contract = deploy(vm, '602a5f555f545f5260205ff3')
    const before = vm.snapshot()
    const call = execute(vm, contract)
    assert.equal(BigInt(call.returnValue), 42n)
    assert.equal(vm.snapshot(), before)
    const tx = execute(vm, contract, '0x', { commit: true })
    assert.equal(tx.gasUsed, 1_000_000)
    assert.ok(tx.gasSpent < tx.gasUsed)
    assert.equal(BigInt(JSON.parse(vm.account(contract)).storage[zero]), 42n)
    restored.restore(vm.snapshot())
    assert.equal(restored.snapshot(), vm.snapshot())
    assert.throws(() => restored.restore('{}'))
    assert.equal(restored.snapshot(), vm.snapshot())
  } finally { vm.free(); restored.free() }
})

test('packed module preserves extension PCs, stack effects and specification gating', () => {
  for (const spec of ['monad-ten', 'monad-next']) {
    const vm = engine(spec)
    try {
      const depth = execute(vm, deploy(vm, 'ae005f5260205ff3'))
      assert.equal(depth.success, spec === 'monad-next')
      assert.equal(depth.structLogs[0].pc, 0)
      assert.equal(depth.structLogs[0].op, spec === 'monad-next' ? 'CALLSTACKDEPTH' : 'INVALID')
      if (spec === 'monad-next') {
        assert.equal(depth.structLogs[1].pc, 2)
        assert.equal(depth.structLogs[1].stack[0], '0'.repeat(64))
        const caller = execute(vm, deploy(vm, '5fae015f5260205ff3'))
        assert.equal(BigInt(caller.returnValue), BigInt(sender))
      }
    } finally { vm.free() }
  }
})

test('execution uses caller-supplied block context', () => {
  const vm = engine()
  try {
    const contract = deploy(vm, '4860005241602052456040524460605260806000f3')
    const result = execute(vm, contract, '0x', {
      baseFee: '0x7', gasPrice: '0x8', beneficiary: sender,
      blockGasLimit: 2_000_000, prevrandao: '0x' + 'ab'.repeat(32)
    })
    assert.equal(result.success, true)
    const words = result.returnValue.slice(2).match(/.{64}/g).map(word => BigInt('0x' + word))
    assert.deepEqual(words, [7n, BigInt(sender), 2_000_000n, BigInt('0x' + 'ab'.repeat(32))])
  } finally { vm.free() }
})

test('package contains source provenance and no runtime build dependency', () => {
  const manifest = require('monad-revmjs/package.json')
  const info = require('monad-revmjs/build-info.json')
  assert.equal(manifest.name, 'monad-revmjs')
  assert.equal(manifest.scripts, undefined)
  assert.equal(manifest.dependencies, undefined)
  assert.match(info.sourceRevision, /^[0-9a-f]{40}$/)
  assert.match(info.cargoLockSha256, /^[0-9a-f]{64}$/)
  assert.equal(info.target, 'wasm32-unknown-unknown')
})
