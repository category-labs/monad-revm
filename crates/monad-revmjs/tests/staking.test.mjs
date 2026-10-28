import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'
import { test } from 'node:test'
import { pathToFileURL } from 'node:url'

const require = createRequire(path.join(process.env.MONAD_REVMJS_TEST_PROJECT, 'package.json'))
const { default: init, Engine } = await import(pathToFileURL(require.resolve('monad-revmjs')))
await init({ module_or_path: readFileSync(require.resolve('monad-revmjs/wasm')) })
const fixtures = JSON.parse(readFileSync(new URL('./fixtures/staking.json', import.meta.url)))
const native = JSON.parse(readFileSync(process.env.MONAD_REVMJS_NATIVE_STAKING))
const zero = '0x' + '0'.repeat(64)
function subset(actual, expected) {
  if (expected !== null && typeof expected === 'object' && !Array.isArray(expected)) {
    for (const [key, value] of Object.entries(expected)) subset(actual[key], value)
  } else assert.deepEqual(actual, expected)
}

const labels = []
for (const fork of ['monad-ten', 'monad-next']) {
  for (const fixture of fixtures.cases) {
    const label = `${fork}/${fixture.name}`
    labels.push(label)
    test(`native/WASM staking parity: ${label}`, () => {
      let vm = new Engine(fork, 1337)
      try {
        const initial = JSON.parse(vm.snapshot())
        initial.accounts = { ...fixtures.accounts, ...fixture.accounts }
        vm.restore(JSON.stringify(initial))
        const results = []
        for (const step of fixture.steps) {
          const system = !!step.request.action
          const request = system
            ? { blockNumber: 1, timestamp: 1_800_000_000, commit: false, ...step.request }
            : { from: fixtures.payer, to: fixtures.staking, data: '0x', value: '0x0', gas: 3_000_000,
                gasPrice: '0x0', blockNumber: 1, timestamp: 1_800_000_000, blockGasLimit: 30_000_000,
                baseFee: '0x0', beneficiary: '0x' + '0'.repeat(40), prevrandao: zero,
                commit: false, trace: true, ...step.request }
          if (!system) request.nonce = JSON.parse(vm.account(request.from)).nonce
          // Match serde_json's sorted object keys, including positions in validation errors.
          const run = request => {
            const json = JSON.stringify(request, (_, value) => value && typeof value === 'object' && !Array.isArray(value)
              ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b))) : value)
            return system ? vm.staking_transition(json) : vm.execute(json)
          }
          const before = vm.snapshot()
          if (step.error) {
            let message
            for (const commit of [false, true]) {
              assert.throws(() => {
                try { run({ ...request, commit }) } catch (error) { message = String(error); throw new Error(message) }
              }, new RegExp(step.error), step.name)
              assert.equal(vm.snapshot(), before)
            }
            results.push({ error: message })
            continue
          }
          const result = JSON.parse(run(request))
          subset(result, step.result)
          assert.equal(result.logs.length, step.logCount, step.name)
          if (!system) {
            const plain = JSON.parse(run({ ...request, trace: false }))
            for (const field of ['success', 'error', 'returnValue', 'gasUsed', 'gasSpent', 'logs']) assert.deepEqual(result[field], plain[field])
          }
          assert.equal(vm.snapshot(), before, step.name)
          assert.deepEqual(JSON.parse(run({ ...request, commit: true })), result)
          if (system && !result.success) assert.equal(vm.snapshot(), before, step.name)
          if (step.stakingUnchanged) assert.deepEqual(JSON.parse(vm.account(fixtures.staking)), JSON.parse(before).accounts[fixtures.staking], step.name)
          for (const [address, expected] of Object.entries(step.accounts || {})) subset(JSON.parse(vm.account(address)), expected)
          const after = vm.snapshot()
          vm.free()
          vm = new Engine(fork, 1337)
          vm.restore(after)
          assert.equal(vm.snapshot(), after)
          results.push({ result, state: JSON.parse(after) })
        }
        assert.deepEqual(results, native[label], 'results, logs, complete traces and restored state match native execution')
      } finally { vm.free() }
    })
  }
}
test('native and WASM staking runners exercise the same cases', () => {
  assert.deepEqual(labels.sort(), Object.keys(native).sort())
})
