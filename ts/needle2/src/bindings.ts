// @ts-expect-error The pinned upstream Emscripten binding does not publish declarations.
import bundledFactory from "../vendor/needle.mjs";
import { defaultAssetURL, Needle2Error } from "./internal.js";
import type { Needle2Factory } from "./internal.js";
import type {
	Needle2CompletionOptions,
	Needle2ResolvedInitialization,
} from "./runtime.js";

export type Needle2NativeGeneration = {
	json: string;
	tokenCount: number;
};

const OUTPUT_CAPACITY = 65_536;

export interface Needle2Binding {
	load(weights: Uint8Array): void;
	initialize(initialization: Needle2ResolvedInitialization): void;
	complete(options: Needle2CompletionOptions): Needle2NativeGeneration;
	reset(): void;
}

export async function wasmBinding(
	wasm: Uint8Array,
	factory: Needle2Factory = bundledFactory as Needle2Factory,
): Promise<Needle2Binding> {
	return Promise.resolve(factory({ wasmBinary: wasm })).then(
		(value) => new WasmBinding(emscriptenModule(value)),
	);
}

export async function nativeBinding(): Promise<Needle2Binding> {
	return loadNativeModule().then(
		(module) => new NativeBinding(module),
	);
}

export type Needle2NativeModule = {
	needleLoad(weights: Uint8Array): number;
	needleInit(
		systemPrompt: string,
		toolsJSON: string,
		toolIndexPath: string | null,
	): number;
	needleComplete(
		prompt: string,
		maxTokens: number,
		outputCapacity: number,
	): Needle2NativeGeneration;
	needleReset(): void;
};

type DenoLike = {
	dlopen(
		path: string,
		symbols: Record<string, unknown>,
	): { symbols: Record<string, (...arguments_: unknown[]) => unknown> };
};

type NativeSymbols = Record<string, (...arguments_: unknown[]) => unknown>;

const denoSymbols = {
	needle_load: { parameters: ["buffer", "u64"], result: "i32" },
	needle_init: {
		parameters: ["buffer", "buffer", "buffer"],
		result: "i32",
	},
	needle_complete: {
		parameters: ["buffer", "i32", "buffer", "i32"],
		result: "i32",
	},
	needle_reset: { parameters: [], result: "void" },
} as const;

let nativeModulePromise: Promise<Needle2NativeModule> | undefined;

function loadNativeModule(): Promise<Needle2NativeModule> {
	if (nativeModulePromise) return nativeModulePromise;
	const promise = loadUncachedNativeModule();
	nativeModulePromise = promise.catch((error) => {
		nativeModulePromise = undefined;
		throw error;
	});
	return nativeModulePromise;
}

async function loadNodeAddon(): Promise<Needle2NativeModule> {
	const moduleSpecifier = "node:module";
	const urlSpecifier = "node:url";
	const { createRequire } = await import(/* @vite-ignore */ moduleSpecifier);
	const { fileURLToPath } = await import(/* @vite-ignore */ urlSpecifier);
	const loadAddon = createRequire(import.meta.url);
	return nativeModule(
		loadAddon(fileURLToPath(defaultAssetURL("native/needle2.node"))),
	);
}

async function loadCAbiLibrary(url: URL): Promise<Needle2NativeModule> {
	const path = decodeURIComponent(url.pathname);
	const deno = (globalThis as { Deno?: DenoLike }).Deno;
	if (deno) return nativeSymbols(deno.dlopen(path, denoSymbols).symbols);

	const bun = (globalThis as { Bun?: object }).Bun;
	if (bun) {
		const ffiSpecifier = "bun:ffi";
		const { dlopen } = await import(/* @vite-ignore */ ffiSpecifier);
		const library = dlopen(path, {
			needle_load: { args: ["ptr", "u64"], returns: "i32" },
			needle_init: {
				args: ["ptr", "ptr", "ptr"],
				returns: "i32",
			},
			needle_complete: {
				args: ["ptr", "i32", "ptr", "i32"],
				returns: "i32",
			},
			needle_reset: { args: [], returns: "void" },
		});
		return nativeSymbols(library.symbols as NativeSymbols);
	}

	throw new Needle2Error(
		"native-unavailable",
		"C ABI native libraries are supported by Deno and Bun only.",
	);
}

function nativeSymbols(symbols: NativeSymbols): Needle2NativeModule {
	return {
		needleLoad(weights) {
			return Number(symbols.needle_load(weights, BigInt(weights.byteLength)));
		},
		needleInit(systemPrompt, toolsJSON, toolIndexPath) {
			return Number(
				symbols.needle_init(
					toCString(systemPrompt),
					toCString(toolsJSON),
					toolIndexPath === null ? null : toCString(toolIndexPath),
				),
			);
		},
		needleComplete(prompt, maxTokens, outputCapacity) {
			const output = new Uint8Array(outputCapacity);
			const tokenCount = Number(
				symbols.needle_complete(
					toCString(prompt),
					maxTokens,
					output,
					outputCapacity,
				),
			);
			return { json: decodeOutput(output), tokenCount };
		},
		needleReset() {
			symbols.needle_reset();
		},
	};
}

function isNodeRuntime(): boolean {
	const process = (
		globalThis as {
			process?: { versions?: { node?: string } };
		}
	).process;
	return process?.versions?.node !== undefined && !isCAbiRuntime();
}

function isCAbiRuntime(): boolean {
	return (
		(globalThis as { Deno?: unknown }).Deno !== undefined ||
		(globalThis as { Bun?: unknown }).Bun !== undefined
	);
}

function nativeLibraryFilename(): string {
	const deno = (globalThis as { Deno?: { build?: { os?: string } } }).Deno;
	const process = (globalThis as { process?: { platform?: string } }).process;
	const platform = deno?.build?.os ?? process?.platform;
	if (platform === "darwin") return "libneedle2.dylib";
	if (platform === "windows" || platform === "win32") return "needle2.dll";
	return "libneedle2.so";
}

function toCString(value: string): Uint8Array {
	const bytes = new TextEncoder().encode(value);
	const result = new Uint8Array(bytes.byteLength + 1);
	result.set(bytes);
	return result;
}

function decodeOutput(output: Uint8Array): string {
	const terminator = output.indexOf(0);
	return new TextDecoder().decode(
		terminator < 0 ? output : output.subarray(0, terminator),
	);
}

type Needle2EmscriptenModule = {
	readonly HEAPU8: Uint8Array;
	readonly _needle_load: (pointer: number, length: bigint) => number;
	readonly _needle_reset: () => void;
	readonly _malloc: (capacity: number) => number;
	readonly _free: (pointer: number) => void;
	readonly UTF8ToString: (pointer: number) => string;
	readonly ccall: (
		name: string,
		returnType: "number",
		argumentTypes: readonly string[],
		argumentValues: readonly unknown[],
	) => number;
};

class WasmBinding implements Needle2Binding {
	private weightsPointer: number | undefined;

	constructor(private readonly module: Needle2EmscriptenModule) {}

	load(weights: Uint8Array): void {
		const pointer = this.module._malloc(weights.byteLength);
		if (pointer === 0) {
			throw new Needle2Error(
				"allocation-failed",
				"Needle 2 could not allocate its weights buffer.",
			);
		}
		this.module.HEAPU8.set(weights, pointer);
		const result = this.module._needle_load(pointer, BigInt(weights.byteLength));
		if (result < 0) {
			this.module._free(pointer);
			throw new Needle2Error(
				"loading-failed",
				`needle_load failed with status ${result}.`,
			);
		}
		if (this.weightsPointer) this.module._free(this.weightsPointer);
		this.weightsPointer = pointer;
	}

	initialize(initialization: Needle2ResolvedInitialization): void {
		const result = this.module.ccall(
			"needle_init",
			"number",
			["string", "string", "string"],
			[
				initialization.systemPrompt,
				JSON.stringify(initialization.tools),
				initialization.toolIndexPath ?? null,
			],
		);
		if (result < 0) {
			throw new Needle2Error(
				"initialization-failed",
				`needle_init failed with status ${result}.`,
			);
		}
	}

	complete(options: Needle2CompletionOptions): Needle2NativeGeneration {
		const outputCapacity = OUTPUT_CAPACITY;
		const output = this.module._malloc(outputCapacity);
		if (output === 0) {
			throw new Needle2Error(
				"allocation-failed",
				"Needle 2 could not allocate its output buffer.",
			);
		}
		try {
			const tokenCount = this.module.ccall(
				"needle_complete",
				"number",
				["string", "number", "number", "number"],
				[options.prompt, options.maxTokens, output, outputCapacity],
			);
			if (tokenCount < 0) {
				throw new Needle2Error(
					"generation-failed",
					`needle_complete failed with status ${tokenCount}.`,
				);
			}
			return {
				json: this.module.UTF8ToString(output),
				tokenCount,
			};
		} finally {
			this.module._free(output);
		}
	}

	reset(): void {
		this.module._needle_reset();
	}
}

class NativeBinding implements Needle2Binding {
	constructor(private readonly module: Needle2NativeModule) {}

	load(weights: Uint8Array): void {
		const result = this.module.needleLoad(weights);
		if (result < 0) {
			throw new Needle2Error(
				"loading-failed",
				`needle_load failed with status ${result}.`,
			);
		}
	}

	initialize(initialization: Needle2ResolvedInitialization): void {
		const result = this.module.needleInit(
			initialization.systemPrompt,
			JSON.stringify(initialization.tools),
			initialization.toolIndexPath ?? null,
		);
		if (result < 0) {
			throw new Needle2Error(
				"initialization-failed",
				`needle_init failed with status ${result}.`,
			);
		}
	}

	complete(options: Needle2CompletionOptions): Needle2NativeGeneration {
		const result = this.module.needleComplete(
			options.prompt,
			options.maxTokens,
			OUTPUT_CAPACITY,
		);
		if (result.tokenCount < 0) {
			throw new Needle2Error(
				"generation-failed",
				`needle_complete failed with status ${result.tokenCount}.`,
			);
		}
		return result;
	}

	reset(): void {
		this.module.needleReset();
	}
}

function emscriptenModule(value: unknown): Needle2EmscriptenModule {
	return value as Needle2EmscriptenModule;
}

function nativeModule(value: unknown): Needle2NativeModule {
	return ((value as { default?: unknown }).default ??
		value) as Needle2NativeModule;
}

function loadUncachedNativeModule(): Promise<Needle2NativeModule> {
	if (isNodeRuntime()) {
		return loadNodeAddon();
	}
	if (isCAbiRuntime()) {
		return loadCAbiLibrary(
			defaultAssetURL(`native/${nativeLibraryFilename()}`),
		);
	}
	return Promise.reject(
		new Needle2Error(
			"native-unavailable",
			"The native Needle 2 engine is unavailable in this runtime.",
		),
	);
}
