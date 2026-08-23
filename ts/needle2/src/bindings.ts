// @ts-expect-error The pinned upstream Emscripten binding does not publish declarations.
import bundledFactory from "../vendor/needle.mjs";
import {
	binarySourceBytes,
	defaultAssetURL,
	Needle2Error,
	PromiseQueue,
} from "./internal.js";
import type { Needle2BinarySource, Needle2Factory } from "./internal.js";
import type {
	Needle2Engine,
	Needle2ResolvedGenerateOptions,
} from "./runtime.js";

export type Needle2NativeGeneration = {
	json: string;
	tokenCount: number;
};

interface Needle2Binding {
	readonly provider: "direct";
	generate(
		options: Needle2ResolvedGenerateOptions,
	): Promise<Needle2NativeGeneration>;
	load(weights: Needle2BinarySource): Promise<void>;
	reset(): Promise<void>;
	dispose(): Promise<void>;
}

export async function createNeedle2Binding(
	engine: Needle2Engine,
	wasm: Needle2BinarySource,
	weights: Needle2BinarySource,
	factory?: Needle2Factory,
): Promise<Needle2Binding> {
	if (engine === "native") {
		return Needle2NativeBinding.create(weights);
	}
	if (engine === "auto") {
		try {
			return await Needle2NativeBinding.create(weights);
		} catch {
			return Needle2WASMBinding.create(wasm, weights, factory);
		}
	}
	return Needle2WASMBinding.create(wasm, weights, factory);
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
	const [{ createRequire }, { fileURLToPath }] = await Promise.all([
		import(/* @vite-ignore */ moduleSpecifier),
		import(/* @vite-ignore */ urlSpecifier),
	]);
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

type Needle2BindingState = {
	disposed: boolean;
	weights: Uint8Array;
	weightsIdentity: object | string;
};

interface Needle2RawBinding {
	loadWeights(weights: Uint8Array): number | undefined;
	initialize(options: Needle2ResolvedGenerateOptions): void;
	complete(options: Needle2ResolvedGenerateOptions): Needle2NativeGeneration;
	reset(): void;
}

class SharedNeedle2Binding {
	private readonly operations = new PromiseQueue();
	private activeOwner: object | undefined;
	private activeInitializationFingerprint: string | undefined;
	private loadedWeightsIdentity: object | string | undefined;

	constructor(private readonly rawBinding: Needle2RawBinding) {}

	load(
		owner: object,
		state: Needle2BindingState,
		weights: Promise<Uint8Array>,
		weightsIdentity: object | string,
	): Promise<void> {
		return this.operations.enqueue(async () => {
			requireActive(state);
			this.requireAvailable(owner);
			if (this.activeOwner === owner) {
				throw new Needle2Error(
					"active-session",
					"Reset the active Needle 2 runtime before loading new weights.",
				);
			}
			const loadedWeights = await weights;
			this.loadWeights(loadedWeights, weightsIdentity);
			state.weights = loadedWeights.slice();
			state.weightsIdentity = weightsIdentity;
		});
	}

	generate(
		owner: object,
		state: Needle2BindingState,
		options: Needle2ResolvedGenerateOptions,
	): Promise<Needle2NativeGeneration> {
		return this.operations.enqueue(() => {
			requireActive(state);
			this.loadWeights(state.weights, state.weightsIdentity);
			const initializationFingerprint = JSON.stringify(options.initialization);
			this.requireAvailable(owner);
			if (this.activeOwner === undefined) {
				this.rawBinding.initialize(options);
				this.activeOwner = owner;
				this.activeInitializationFingerprint = initializationFingerprint;
			} else if (
				this.activeInitializationFingerprint !== initializationFingerprint
			) {
				throw new Needle2Error(
					"initialization-changed",
					"Reset the Needle 2 runtime before changing its tools, system facts, or tool index.",
				);
			}
			return this.rawBinding.complete(options);
		});
	}

	reset(owner: object, state: Needle2BindingState): Promise<void> {
		return this.operations.enqueue(() => {
			requireActive(state);
			this.requireAvailable(owner);
			if (this.activeOwner === owner) {
				this.rawBinding.reset();
				this.clearActiveConversation();
			}
		});
	}

	dispose(owner: object, state: Needle2BindingState): Promise<void> {
		return this.operations.enqueue(() => {
			if (this.activeOwner === owner) {
				this.rawBinding.reset();
				this.clearActiveConversation();
			}
			state.disposed = true;
		});
	}

	private requireAvailable(owner: object): void {
		if (this.activeOwner !== undefined && this.activeOwner !== owner) {
			throw new Needle2Error(
				"active-session",
				"Another direct Needle 2 runtime has an active conversation. Reset or dispose it before using this runtime.",
			);
		}
	}

	private clearActiveConversation(): void {
		this.activeOwner = undefined;
		this.activeInitializationFingerprint = undefined;
	}

	private loadWeights(weights: Uint8Array, identity: object | string): void {
		if (identity === this.loadedWeightsIdentity) return;

		const result = this.rawBinding.loadWeights(weights);
		if (result !== undefined && result < 0) {
			throw new Needle2Error(
				"loading-failed",
				`needle_load failed with status ${result}.`,
			);
		}
		this.loadedWeightsIdentity = identity;
		this.activeInitializationFingerprint = undefined;
	}
}

class ManagedNeedle2Binding implements Needle2Binding {
	readonly provider = "direct" as const;
	private readonly state: Needle2BindingState;

	protected constructor(
		private readonly sharedBinding: SharedNeedle2Binding,
		weights: Uint8Array,
		weightsIdentity: object | string,
	) {
		this.state = {
			disposed: false,
			weights,
			weightsIdentity,
		};
	}

	generate(
		options: Needle2ResolvedGenerateOptions,
	): Promise<Needle2NativeGeneration> {
		return this.sharedBinding.generate(this, this.state, options);
	}

	load(weights: Needle2BinarySource): Promise<void> {
		return this.sharedBinding.load(
			this,
			this.state,
			binarySourceBytes(weights),
			{},
		);
	}

	reset(): Promise<void> {
		return this.sharedBinding.reset(this, this.state);
	}

	dispose(): Promise<void> {
		return this.sharedBinding.dispose(this, this.state);
	}

	protected loadInitial(
		weights: Uint8Array,
		weightsIdentity: object | string,
	): Promise<void> {
		return this.sharedBinding.load(
			this,
			this.state,
			Promise.resolve(weights),
			weightsIdentity,
		);
	}
}

class Needle2WASMBinding extends ManagedNeedle2Binding {
	static async create(
		wasm: Needle2BinarySource,
		weights: Needle2BinarySource,
		factory: Needle2Factory = bundledFactory as Needle2Factory,
	): Promise<Needle2WASMBinding> {
		const [wasmBytes, weightBytes] = await Promise.all([
			binarySourceBytes(wasm),
			binarySourceBytes(weights),
		]);
		const sharedBinding = await createWasmBinding(factory, wasmBytes);
		const weightsIdentity = binarySourceIdentity(weights);
		const binding = new Needle2WASMBinding(
			sharedBinding,
			weightBytes,
			weightsIdentity,
		);
		await binding.loadInitial(weightBytes, weightsIdentity);
		return binding;
	}
}

class Needle2NativeBinding extends ManagedNeedle2Binding {
	static async create(
		weights: Needle2BinarySource,
	): Promise<Needle2NativeBinding> {
		const module = await loadNativeModule();
		const weightBytes = await binarySourceBytes(weights);
		const sharedBinding = sharedNativeBinding(module);
		const weightsIdentity = binarySourceIdentity(weights);
		const binding = new Needle2NativeBinding(
			sharedBinding,
			weightBytes,
			weightsIdentity,
		);
		await binding.loadInitial(weightBytes, weightsIdentity);
		return binding;
	}
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

class WasmRawBinding implements Needle2RawBinding {
	private weightsPointer: number | undefined;

	constructor(private readonly module: Needle2EmscriptenModule) {}

	loadWeights(weights: Uint8Array): number | undefined {
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
			return result;
		}
		if (this.weightsPointer) this.module._free(this.weightsPointer);
		this.weightsPointer = pointer;
	}

	initialize(options: Needle2ResolvedGenerateOptions): void {
		const initialization = options.initialization;
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

	complete(options: Needle2ResolvedGenerateOptions): Needle2NativeGeneration {
		const maxTokens = positiveInteger(options.maxTokens ?? 256, "maxTokens");
		const outputCapacity = positiveInteger(
			options.outputCapacity ?? 65_536,
			"outputCapacity",
		);
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
				[options.prompt, maxTokens, output, outputCapacity],
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

let sharedWasmModule: Promise<SharedNeedle2Binding> | undefined;
let sharedWasmFactory: Needle2Factory | undefined;

async function createWasmBinding(
	factory: Needle2Factory,
	wasm: Uint8Array,
): Promise<SharedNeedle2Binding> {
	if (sharedWasmModule && sharedWasmFactory === factory) {
		return sharedWasmModule;
	}
	sharedWasmFactory = factory;
	const modulePromise = Promise.resolve(factory({ wasmBinary: wasm })).then(
		(value) =>
			new SharedNeedle2Binding(new WasmRawBinding(emscriptenModule(value))),
	);
	const cachedPromise = modulePromise.catch((error) => {
		if (sharedWasmModule === cachedPromise) {
			sharedWasmModule = undefined;
			sharedWasmFactory = undefined;
		}
		throw error;
	});
	sharedWasmModule = cachedPromise;
	return cachedPromise;
}

let sharedNativeBindingInstance: SharedNeedle2Binding | undefined;

function sharedNativeBinding(
	module: Needle2NativeModule,
): SharedNeedle2Binding {
	sharedNativeBindingInstance ??= new SharedNeedle2Binding(
		new NativeRawBinding(module),
	);
	return sharedNativeBindingInstance;
}

class NativeRawBinding implements Needle2RawBinding {
	constructor(private readonly module: Needle2NativeModule) {}

	loadWeights(weights: Uint8Array): number | undefined {
		const result = this.module.needleLoad(weights);
		return result < 0 ? result : undefined;
	}

	initialize(options: Needle2ResolvedGenerateOptions): void {
		const initialization = options.initialization;
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

	complete(options: Needle2ResolvedGenerateOptions): Needle2NativeGeneration {
		const result = this.module.needleComplete(
			options.prompt,
			positiveInteger(options.maxTokens ?? 256, "maxTokens"),
			positiveInteger(options.outputCapacity ?? 65_536, "outputCapacity"),
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
	if (typeof value !== "object" || value === null) throw invalidModule();

	const module = value as Partial<Needle2EmscriptenModule>;
	if (
		!(module.HEAPU8 instanceof Uint8Array) ||
		typeof module._needle_load !== "function" ||
		typeof module._needle_reset !== "function" ||
		typeof module._malloc !== "function" ||
		typeof module._free !== "function" ||
		typeof module.UTF8ToString !== "function" ||
		typeof module.ccall !== "function"
	) {
		throw invalidModule();
	}
	return module as Needle2EmscriptenModule;
}

function invalidModule(): Needle2Error {
	return new Needle2Error(
		"invalid-module",
		"The Needle 2 factory returned an object without the expected Emscripten exports.",
	);
}

function nativeModule(value: unknown): Needle2NativeModule {
	if (typeof value !== "object" || value === null) {
		throw new Needle2Error(
			"invalid-native-module",
			"The Needle 2 native addon did not return an object.",
		);
	}
	const candidate = (value as { default?: unknown }).default ?? value;
	if (typeof candidate !== "object" || candidate === null) {
		throw new Needle2Error(
			"invalid-native-module",
			"The Needle 2 native addon did not return an object.",
		);
	}
	const module = candidate as Partial<Needle2NativeModule>;
	if (
		typeof module.needleLoad !== "function" ||
		typeof module.needleInit !== "function" ||
		typeof module.needleComplete !== "function" ||
		typeof module.needleReset !== "function"
	) {
		throw new Needle2Error(
			"invalid-native-module",
			"The Needle 2 native addon is missing the expected exports.",
		);
	}
	return module as Needle2NativeModule;
}

function positiveInteger(value: number, name: string): number {
	if (!Number.isSafeInteger(value) || value <= 0 || value > 2_147_483_647) {
		throw new Needle2Error(
			"invalid-generate-options",
			`Needle 2 ${name} must be an integer between 1 and 2147483647.`,
		);
	}
	return value;
}

function binarySourceIdentity(source: Needle2BinarySource): object | string {
	if (typeof source === "string") return source;
	if (source instanceof URL) return source.href;
	return source;
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

function requireActive(state: Needle2BindingState): void {
	if (state.disposed) {
		throw new Needle2Error(
			"disposed",
			"This Needle 2 runtime has been disposed.",
		);
	}
}
