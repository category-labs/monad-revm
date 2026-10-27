import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { test } from 'node:test'
import { pathToFileURL } from 'node:url'

const require = createRequire(path.join(process.env.MONAD_REVMJS_TEST_PROJECT, 'package.json'))
const { default: init, Engine } = await import(pathToFileURL(require.resolve('monad-revmjs')))
await init({ module_or_path: readFileSync(require.resolve('monad-revmjs/wasm')) })
const fixtures = JSON.parse(readFileSync(new URL('./fixtures/transactions.json', import.meta.url)))
const native = JSON.parse(readFileSync(process.env.MONAD_REVMJS_NATIVE_TRANSACTIONS))
const zero = '0x' + '0'.repeat(64)
function subset(actual, expected) {
  if (expected !== null && typeof expected === 'object') {
    for (const [key, value] of Object.entries(expected)) subset(actual[key], value)
  } else assert.deepEqual(actual, expected)
}

const labels = []
for (const fork of ['monad-ten', 'monad-next']) {
  for (const fixture of fixtures.cases) {
    const label = `${fork}/${fixture.name}`
    labels.push(label)
    test(`native/WASM typed-transaction parity: ${label}`, () => {
      const vm = new Engine(fork, 1337)
      try {
        const snapshot = JSON.parse(vm.snapshot())
        for (const accounts of [fixtures.accounts, fixture.accounts]) {
          for (const [address, fields] of Object.entries(accounts)) {
            snapshot.accounts[address] = { balance: '0x0', nonce: 0, code: '0x', storage: {}, ...snapshot.accounts[address], ...fields }
          }
        }
        vm.restore(JSON.stringify(snapshot))
        const results = []
        for (const step of fixture.steps) {
          const request = {
            from: fixtures.payer, to: fixtures.authority, data: '0x', value: '0x0', gas: 100_000,
            gasPrice: '0x0', blockNumber: 1, timestamp: 1_800_000_000, blockGasLimit: 30_000_000,
            baseFee: '0x0', beneficiary: '0x' + '0'.repeat(40), prevrandao: zero,
            commit: false, trace: true, ...step.request
          }
          request.nonce = JSON.parse(vm.account(request.from)).nonce
          const before = vm.snapshot()
          if (step.error) {
            let message
            assert.throws(() => {
              try { vm.execute(JSON.stringify(request)) } catch (error) { message = String(error); throw new Error(message) }
            }, new RegExp(step.error))
            assert.equal(vm.snapshot(), before)
            results.push({ error: message })
            continue
          }
          const result = JSON.parse(vm.execute(JSON.stringify(request)))
          subset(result, step.result)
          assert.equal(result.gasUsed, request.gas)
          for (const [opcode, cost] of Object.entries(step.opcodeCosts || {})) {
            assert.equal(result.structLogs.find(entry => entry.op === opcode).gasCost, cost)
          }
          const plain = JSON.parse(vm.execute(JSON.stringify({ ...request, trace: false })))
          for (const field of ['success', 'error', 'returnValue', 'gasUsed', 'gasSpent', 'logs', 'authorizationAuthorities']) assert.deepEqual(result[field], plain[field])
          assert.equal(vm.snapshot(), before)
          assert.deepEqual(JSON.parse(vm.execute(JSON.stringify({ ...request, commit: true }))), result)
          for (const [address, expected] of Object.entries(step.accounts)) {
            const actual = JSON.parse(vm.account(address))
            for (const key of Object.keys(expected.storage || {})) actual.storage[key] ??= zero
            subset(actual, expected)
          }
          const after = vm.snapshot()
          const restored = new Engine(fork, 1337)
          try { restored.restore(after); assert.equal(restored.snapshot(), after) } finally { restored.free() }
          results.push({ result, state: JSON.parse(after) })
        }
        assert.deepEqual(results, native[label], 'complete results, traces and state match native execution')
      } finally { vm.free() }
    })
  }
}
test('native and WASM typed-transaction runners exercise the same cases', () => {
  assert.deepEqual(labels.sort(), Object.keys(native).sort())
})
