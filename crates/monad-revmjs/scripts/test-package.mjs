import { spawnSync } from 'node:child_process'
import { closeSync, mkdtempSync, openSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const crate = fileURLToPath(new URL('..', import.meta.url))
const output = path.resolve(crate, '../../target/monad-revmjs')
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm'
function run(command, args, options = {}) {
  const result = spawnSync(command, args, { stdio: 'inherit', ...options })
  if (result.error) throw result.error
  if (result.status !== 0) throw new Error(`${command} failed (${result.status})`)
  return result
}

const packed = run(npm, ['pack', '--ignore-scripts', '--json', '--pack-destination', output], {
  cwd: path.join(output, 'package'), stdio: 'pipe', encoding: 'utf8'
})
const [archive] = JSON.parse(packed.stdout)
const project = mkdtempSync(path.join(tmpdir(), 'monad-revmjs-package-'))
try {
  writeFileSync(path.join(project, 'package.json'), '{"private":true,"type":"module"}\n')
  run(npm, ['install', '--ignore-scripts', '--no-audit', '--no-fund', path.join(output, archive.filename)], { cwd: project })
  const installed = JSON.parse(readFileSync(path.join(project, 'node_modules/monad-revmjs/package.json'), 'utf8'))
  if (installed.scripts || installed.dependencies) throw new Error('The runtime package must be self-contained with no install scripts')
  const nativeResults = path.join(project, 'native-precompiles.json')
  const nativeOutput = openSync(nativeResults, 'w')
  try {
    // Enable the normal native crypto backends in this separate host build.
    // The browser artifact was built with the wrapper's pure-Rust dependencies.
    run('cargo', ['+1.91', 'run', '--locked', '-p', 'monad-revmjs', '--example', 'precompile_results', '--features', 'monad-revm/default'], {
      cwd: path.resolve(crate, '../..'), stdio: ['ignore', nativeOutput, 'inherit']
    })
  } finally { closeSync(nativeOutput) }
  const nativeTransactions = path.join(project, 'native-transactions.json')
  const transactionOutput = openSync(nativeTransactions, 'w')
  try {
    run('cargo', ['+1.91', 'run', '--locked', '-p', 'monad-revmjs', '--example', 'transaction_results', '--features', 'monad-revm/default'], {
      cwd: path.resolve(crate, '../..'), stdio: ['ignore', transactionOutput, 'inherit']
    })
  } finally { closeSync(transactionOutput) }
  const nativeStaking = path.join(project, 'native-staking.json')
  const stakingOutput = openSync(nativeStaking, 'w')
  try {
    run('cargo', ['+1.91', 'run', '--locked', '-p', 'monad-revmjs', '--example', 'staking_results', '--features', 'monad-revm/default'], {
      cwd: path.resolve(crate, '../..'), stdio: ['ignore', stakingOutput, 'inherit']
    })
  } finally { closeSync(stakingOutput) }
  run(process.execPath, ['--test', ...['package', 'precompiles', 'transactions', 'staking'].map(name => path.join(crate, `tests/${name}.test.mjs`))], {
    env: { ...process.env, MONAD_REVMJS_TEST_PROJECT: project, MONAD_REVMJS_NATIVE_RESULTS: nativeResults, MONAD_REVMJS_NATIVE_TRANSACTIONS: nativeTransactions, MONAD_REVMJS_NATIVE_STAKING: nativeStaking }
  })
  console.log(`Verified installable package: ${path.join(output, archive.filename)}`)
} finally {
  rmSync(project, { recursive: true, force: true })
}
