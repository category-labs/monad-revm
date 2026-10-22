import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
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
  run(process.execPath, ['--test', path.join(crate, 'tests/package.test.mjs')], {
    env: { ...process.env, MONAD_REVMJS_TEST_PROJECT: project }
  })
  console.log(`Verified installable package: ${path.join(output, archive.filename)}`)
} finally {
  rmSync(project, { recursive: true, force: true })
}
