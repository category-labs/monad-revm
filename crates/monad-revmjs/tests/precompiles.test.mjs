import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { test } from 'node:test'
import { pathToFileURL } from 'node:url'

const require = createRequire(path.join(process.env.MONAD_REVMJS_TEST_PROJECT, 'package.json'))
const { default: init, Engine } = await import(pathToFileURL(require.resolve('monad-revmjs')))
await init({ module_or_path: readFileSync(require.resolve('monad-revmjs/wasm')) })
const vectors = JSON.parse(readFileSync(new URL('./fixtures/precompiles.json', import.meta.url)))
const native = JSON.parse(readFileSync(process.env.MONAD_REVMJS_NATIVE_RESULTS))
const sender = '0x1111111111111111111111111111111111111111'
const txGas = 3_000_000

function execute(vm, to, data, commit, trace) {
  return JSON.parse(vm.execute(JSON.stringify({
    from: sender, to, data, value: '0x0', gas: txGas, gasPrice: '0x1',
    nonce: JSON.parse(vm.account(sender)).nonce, blockNumber: 1, timestamp: 1_800_000_000,
    blockGasLimit: 30_000_000, baseFee: '0x0', beneficiary: '0x' + '00'.repeat(20),
    prevrandao: '0x' + '00'.repeat(32), commit, trace
  })))
}

const labels = []
for (const fork of ['monad-ten', 'monad-next']) {
  for (const vector of vectors) {
    const variants = [['exact', vector.gas, vector.success, vector.output]]
    if (vector.success) variants.push(['surplus', vector.gas + 123, true, vector.output], ['out-of-gas', vector.gas - 1, false, '0x'])
    for (const [variant, gasLimit, success, output] of variants) {
      const label = `${fork}/${vector.name}/${variant}`
      labels.push(label)
      test(`native/WASM precompile parity: ${label}`, () => {
        const vm = new Engine(fork, 1337)
        try {
          vm.fund(sender, '0x3635c9adc5dea00000')
          const runtime = `365f5f375f5f365f61${vector.address.toString(16).padStart(4, '0')}63${gasLimit.toString(16).padStart(8, '0')}fa5f523d5f60203e3d6020015ff3`
          const len = (runtime.length / 2).toString(16).padStart(2, '0')
          const deployment = execute(vm, null, `0x60${len}600c60003960${len}6000f3${runtime}`, true, false)
          assert.equal(deployment.success, true)
          const address = deployment.createdAddress
          const before = vm.snapshot()
          const result = execute(vm, address, vector.input, false, true)
          assert.equal(result.success, true)
          assert.equal(result.returnValue, '0x' + Number(success).toString(16).padStart(64, '0') + output.slice(2))
          assert.equal(result.gasUsed, txGas)
          assert.equal(result.traceTruncated, false)
          assert.ok(result.structLogs.every(step => step.depth === 1))
          const call = result.structLogs.findIndex(step => step.op === 'STATICCALL')
          assert.notEqual(call, -1)
          assert.equal(result.structLogs[call].gas - result.structLogs[call + 1].gas, 100 + (success ? vector.gas : gasLimit))
          const plain = execute(vm, address, vector.input, false, false)
          for (const field of ['success', 'error', 'returnValue', 'gasUsed', 'gasSpent', 'logs']) assert.deepEqual(result[field], plain[field])
          assert.equal(vm.snapshot(), before, 'simulations do not mutate state')
          assert.deepEqual(execute(vm, address, vector.input, true, true), result)
          const account = JSON.parse(vm.account(sender))
          assert.equal(account.nonce, 2)
          assert.deepEqual({ execution: result, sender: account }, native[label], 'native default crypto and packaged WASM agree, including the complete trace')
        } finally { vm.free() }
      })
    }
  }
}
test('native and WASM runners exercised the same cases', () => {
  assert.deepEqual(labels.sort(), Object.keys(native).sort())
})
