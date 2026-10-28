export { default, Engine, initSync } from './dist/monad_revmjs.js'
export type { InitInput, InitOutput, SyncInitInput } from './dist/monad_revmjs.js'

export type MonadSpec = 'monad-ten' | 'monad-next'

/** Explicit local lifecycle operations; the host decides when they occur. */
export type StakingAction = { kind: 'snapshot' } | { kind: 'epochChange'; epoch: string } | { kind: 'reward'; author: string; amount: string }

/** JSON input to Engine.staking_transition. Epoch and amount are exact hex quantities. */
export interface StakingRequest {
  action: StakingAction
  blockNumber: number
  timestamp: number
  commit: boolean
}

/** System operations have no user nonce, transaction fee or EVM instruction trace. */
export interface StakingResult {
  success: boolean
  error: string | null
  returnValue: string
  gasSpent: number
  logs: { address: string; topics: string[]; data: string }[]
}

export interface AccessListEntry {
  address: string
  storageKeys: string[]
}

/** Signed authorization; chainId, nonce, r and s are exact hex quantities. */
export interface AuthorizationEntry {
  chainId: string
  address: string
  nonce: string
  yParity: number
  r: string
  s: string
}

/** JSON input to Engine.execute. Integers represented as numbers must be safe integers. */
export interface ExecuteRequest {
  /** Omitted types are inferred from authorization, fee and access-list fields. */
  type?: 0 | 1 | 2 | 4
  accessList?: AccessListEntry[]
  authorizationList?: AuthorizationEntry[]
  /** Block history. Current-block arrays describe preceding transactions only. */
  chainContext?: {
    parentSendersAndAuthorities?: string[]
    grandparentSendersAndAuthorities?: string[]
    currentBlockSenders?: string[]
    currentBlockAuthorities?: string[][]
  }
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
  /** Recoverable authorities, including authorizations skipped for chain ID or nonce. */
  authorizationAuthorities: string[]
  /** Executed runtime code, with delegation resolved; empty when tracing is disabled. */
  codeByAddress: Record<string, string>
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
