export { default, Engine, initSync } from './dist/monad_revmjs.js'
export type { InitInput, InitOutput, SyncInitInput } from './dist/monad_revmjs.js'

export type MonadSpec = 'monad-ten' | 'monad-next'

/** JSON input to Engine.execute. Integers represented as numbers must be safe integers. */
export interface ExecuteRequest {
  from: string
  to: string | null
  data: string
  value: string
  gas: number
  gasPrice: string
  priorityFee?: string | null
  nonce: number
  blockNumber: number
  timestamp: number
  blockGasLimit: number
  baseFee: string
  beneficiary: string
  prevrandao: string
  commit: boolean
  trace: boolean
}

/** JSON output from Engine.account. Addresses, balances, code and storage use 0x hex. */
export interface AccountState {
  balance: string
  nonce: number
  code: string
  storage: Record<string, string>
}

/** State immediately before an instruction; root depth is 1, memory and stack are unprefixed hex. */
export interface TraceStep {
  pc: number
  op: string
  gas: number
  gasCost: number
  depth: number
  stack: string[]
  memory: string
  error?: string
}

/** JSON output from Engine.execute. Validation failures throw instead of returning a result. */
export interface ExecutionResult {
  success: boolean
  error: string | null
  returnValue: string
  createdAddress: string | null
  /** Full transaction gas limit charged by Monad. */
  gasUsed: number
  /** Gas actually spent, including intrinsic gas. */
  gasSpent: number
  logs: { address: string, topics: string[], data: string }[]
  /** Raw 0x calldata sent by STATICCALL to the Hardhat console address when trace is enabled.
   * Preserved across reverts and independently of instruction-trace truncation. */
  consoleCalls: string[]
  structLogs: TraceStep[]
  /** Successful KECCAK256 inputs keyed by hash, both unprefixed hex; shares the trace budget. */
  preimages: Record<string, string>
  traceTruncated: boolean
}
