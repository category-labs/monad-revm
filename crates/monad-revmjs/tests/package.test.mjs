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

test('packed module captures console calls across reverts and trace truncation', () => {
  const consoleCall = '365f5f375f5f365f73000000000000000000636f6e736f6c652e6c6f675afa50'
  for (const spec of ['monad-ten', 'monad-next']) {
    const vm = engine(spec)
    try {
      for (const [runtime, success, truncated] of [
        [consoleCall + '60ff5f5300', true, false],
        [consoleCall + '5f5ffd', false, false],
        ['614e205b600190038060035750' + consoleCall + '00', true, true]
      ]) {
        const contract = deploy(vm, runtime)
        const before = vm.snapshot()
        const result = execute(vm, contract, '0x11223344')
        assert.equal(result.success, success)
        assert.equal(result.traceTruncated, truncated)
        assert.deepEqual(result.consoleCalls, ['0x11223344'])
        assert.equal(vm.snapshot(), before)
      }
    } finally { vm.free() }
  }
})

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

test('missing block hashes are zero and supplied hashes survive restoration', () => {
  for (const spec of ['monad-ten', 'monad-next']) {
    const vm = engine(spec)
    const restored = engine(spec)
    try {
      const contract = deploy(vm, '6001405f5260205ff3')
      const supplied = '0x' + 'ab'.repeat(32)
      for (const expected of [zero, supplied]) {
        if (expected === supplied) vm.set_block_hash(1, supplied)
        restored.restore(vm.snapshot())
        for (const state of [vm, restored]) {
          for (const trace of [false, true]) {
            const result = execute(state, contract, '0x', { blockNumber: 2, trace })
            assert.equal(result.success, true)
            assert.equal(result.returnValue, expected)
          }
          for (const blockNumber of [1, 258]) {
            assert.equal(execute(state, contract, '0x', { blockNumber }).returnValue, zero)
          }
        }
      }
    } finally { vm.free(); restored.free() }
  }
})

test('invalid snapshot bytecode throws a recoverable error and preserves the engine', () => {
  const vm = engine()
  try {
    const contract = deploy(vm, '602a5f555f545f5260205ff3')
    execute(vm, contract, '0x', { commit: true })
    vm.set_block_hash(1, '0x' + 'ab'.repeat(32))
    const before = vm.snapshot()
    for (const code of ['0xef01', '0xef0101' + '00'.repeat(20)]) {
      const invalid = JSON.parse(before)
      invalid.accounts[sender].balance = '0x2'
      invalid.accounts[contract].code = code
      assert.throws(() => vm.restore(JSON.stringify(invalid)), error => {
        assert.ok(!(error instanceof WebAssembly.RuntimeError))
        assert.match(String(error), /Invalid bytecode/)
        return true
      })
      assert.equal(vm.snapshot(), before)
      const result = execute(vm, contract)
      assert.equal(result.success, true)
      assert.equal(BigInt(result.returnValue), 42n)
      assert.equal(vm.snapshot(), before)
    }
  } finally { vm.free() }
})

test('hash preimages include expanded memory and zero-length inputs', () => {
  const vm = engine()
  try {
    const cases = [
      ['60205f20', '00'.repeat(32), ''],
      ['60115f5360405f20', '11' + '00'.repeat(63), '11' + '00'.repeat(31)],
      ['5f60ff20', '', ''],
      ['5f7f' + 'ff'.repeat(32) + '20', '', '']
    ]
    for (const [runtime, input, memoryBefore] of cases) {
      const contract = deploy(vm, runtime + '5f5260205ff3')
      const result = execute(vm, contract)
      assert.equal(result.success, true)
      assert.deepEqual(result.preimages, { [result.returnValue.slice(2)]: input })
      assert.equal(result.traceTruncated, false)
      const step = result.structLogs.find(step => step.op === 'KECCAK256')
      assert.equal(step.memory, memoryBefore)
      assert.equal(step.stack.length, 2)
    }
  } finally { vm.free() }
})

test('hashes that run out of gas do not record preimages', () => {
  const vm = engine()
  try {
    const contract = deploy(vm, '60205f5260205f2000')
    const result = execute(vm, contract, '0x', { gas: 21_020 })
    assert.equal(result.success, false)
    assert.deepEqual(result.preimages, {})
    assert.equal(result.structLogs.at(-1).op, 'KECCAK256')
    assert.match(result.structLogs.at(-1).error, /OutOfGas/)
    assert.equal(result.traceTruncated, false)
  } finally { vm.free() }
})

test('preimages share the trace budget without changing execution', () => {
  const vm = engine()
  try {
    const cases = [
      ['628000005f205f00', 3, 1],
      ['5f620fffff53' + '5b'.repeat(8) + '628000005f205f00', 14, 0]
    ]
    for (const [runtime, stepCount, preimageCount] of cases) {
      const contract = deploy(vm, runtime)
      const traced = execute(vm, contract, '0x', { gas: 3_000_000 })
      const untraced = execute(vm, contract, '0x', { gas: 3_000_000, trace: false })
      assert.equal(traced.success, true)
      assert.equal(traced.gasSpent, untraced.gasSpent)
      assert.equal(traced.returnValue, untraced.returnValue)
      assert.equal(traced.traceTruncated, true)
      assert.equal(traced.structLogs.length, stepCount)
      assert.equal(traced.structLogs.at(-1).op, 'KECCAK256')
      const preimages = Object.values(traced.preimages)
      assert.equal(preimages.length, preimageCount)
      for (const input of preimages) assert.equal(input.length, 16 * 1024 * 1024)
      assert.ok(JSON.stringify(traced).length < 32 * 1024 * 1024)
    }
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
