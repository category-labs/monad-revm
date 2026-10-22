/* tslint:disable */
/* eslint-disable */

/**
 * Owns local state; simulation and tracing do not mutate that state.
 */
export class Engine {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Read an account as JSON, including its bytecode and nonzero storage slots.
     */
    account(account: string): string;
    /**
     * Execute against an isolated cache, applying state only for committed transactions.
     */
    execute(request: string): string;
    /**
     * Set a development account's balance without replacing its code or nonce.
     */
    fund(account: string, balance: string): void;
    /**
     * Create an empty local VM with an explicit specification and chain ID.
     */
    constructor(spec: string, chain_id: number);
    /**
     * Validate completely before replacing the current database.
     */
    restore(snapshot: string): void;
    /**
     * Supply a historical hash for the BLOCKHASH instruction.
     * Unsupplied hashes return zero, including after restoring a snapshot.
     */
    set_block_hash(number: number, hash: string): void;
    /**
     * Export versioned JSON state, independent of any browser persistence mechanism.
     */
    snapshot(): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_engine_free: (a: number, b: number) => void;
    readonly engine_account: (a: number, b: number, c: number, d: number) => void;
    readonly engine_execute: (a: number, b: number, c: number, d: number) => void;
    readonly engine_fund: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
    readonly engine_new: (a: number, b: number, c: number, d: number) => void;
    readonly engine_restore: (a: number, b: number, c: number, d: number) => void;
    readonly engine_set_block_hash: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly engine_snapshot: (a: number, b: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
