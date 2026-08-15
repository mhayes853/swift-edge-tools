import { Needle2DirectBackend, Needle2WorkerBackend } from "./backend.js";
import { Needle2NativeBinding, defaultNativeFactory } from "./bindings.js";
import {
	defaultAssetURL,
	Needle2Error,
	Needle2ProtocolError,
	setAssetBaseURL,
} from "./internal.js";
import { defaultSystemPrompt, type Needle2SystemValues } from "./system.js";
import type { Needle2JSONObject } from "./value.js";
import type { Needle2Backend } from "./backend.js";
import type {
	Needle2BinarySource,
	Needle2Factory,
	Needle2WorkerOptions,
} from "./internal.js";

export type Needle2Provider = "direct" | "worker";
export type Needle2Engine = "wasm" | "native" | "auto";

export type Needle2RuntimeOptions =
	| {
			provider: "direct";
			wasm?: Needle2BinarySource;
			weights?: Needle2BinarySource;
			factory?: Needle2Factory;
			engine?: Needle2Engine;
	  }
	| {
			provider: "worker";
			wasm?: Needle2BinarySource;
			weights?: Needle2BinarySource;
			workerOptions?: Needle2WorkerOptions;
			engine?: Needle2Engine;
	  };

export type Needle2ToolDefinition = {
	name: string;
	description?: string;
	parameters: Needle2JSONObject;
};

export type Needle2Initialization = {
	systemValues?: Needle2SystemValues;
	tools: readonly Needle2ToolDefinition[];
	toolIndexPath?: string;
};

export type Needle2ResolvedInitialization = {
	systemPrompt: string;
	tools: readonly Needle2ToolDefinition[];
	toolIndexPath?: string;
};

export type Needle2GenerateOptions = {
	prompt: string;
	initialization: Needle2Initialization;
	maxTokens?: number;
	outputCapacity?: number;
};

export type Needle2ResolvedGenerateOptions = {
	prompt: string;
	initialization: Needle2ResolvedInitialization;
	maxTokens?: number;
	outputCapacity?: number;
};

export type Needle2FunctionCall = {
	name: string;
	arguments: Needle2JSONObject;
};

export type Needle2GenerationMetrics = {
	prefillTokensPerSecond?: number;
	decodeTokensPerSecond?: number;
	peakRAMMegabytes?: number;
};

export type Needle2ResponseType =
	| "call"
	| "respond"
	| "refuse"
	| "text"
	| "error"
	| (string & {});

export type Needle2GenerationSuccess = {
	success: true;
	type: Needle2ResponseType;
	functionCalls: Needle2FunctionCall[];
	reasoning?: string;
	confidence?: number;
	tokenCount: number;
	metrics: Needle2GenerationMetrics;
};

export type Needle2GenerationFailure = {
	success: false;
	type: Needle2ResponseType;
	error: string;
	errorCode?: string;
	functionCalls: Needle2FunctionCall[];
	reasoning?: string;
	confidence?: number;
	tokenCount: number;
	metrics: Needle2GenerationMetrics;
};

export type Needle2GenerationResult =
	| Needle2GenerationSuccess
	| Needle2GenerationFailure;

try {
	setAssetBaseURL(new URL(import.meta.url));
} catch (cause) {
	throw new Error("Needle 2 could not determine its module URL.", { cause });
}

export class Needle2Runtime {
	readonly provider: Needle2Provider;

	private constructor(private readonly backend: Needle2Backend) {
		this.provider = backend.provider;
	}

	/** @internal */
	static async create(options: Needle2RuntimeOptions): Promise<Needle2Runtime> {
		const weights = options.weights ?? defaultAssetURL("needle2.cact");
		const engine = options.engine ?? "wasm";
		if (options.provider === "direct") {
			if (engine === "native") {
				requireNativeFactory();
				return new Needle2Runtime(await Needle2NativeBinding.create(weights));
			}
			if (engine === "auto") {
				const nativeBinding =
					await Needle2NativeBinding.createIfAvailable(weights);
				if (nativeBinding) {
					return new Needle2Runtime(nativeBinding);
				}
			}

			const wasm = options.wasm ?? defaultAssetURL("needle.wasm");
			return new Needle2Runtime(
				await Needle2DirectBackend.create(wasm, weights, options.factory),
			);
		}

		if (engine === "native") {
			requireNativeFactory();
		}
		const wasm = options.wasm ?? defaultAssetURL("needle.wasm");
		return new Needle2Runtime(
			await Needle2WorkerBackend.create(
				wasm,
				weights,
				options.workerOptions,
				engine,
			),
		);
	}

	async generate(
		options: Needle2GenerateOptions,
	): Promise<Needle2GenerationResult> {
		const generation = await this.backend.generate(
			resolveGenerateOptions(options),
		);
		return parseGenerationResult(generation.json, generation.tokenCount);
	}

	load(weights: Needle2BinarySource): Promise<void> {
		return this.backend.load(weights);
	}

	dispose(): Promise<void> {
		return this.backend.dispose();
	}
}

export function needle2Runtime(
	options: Needle2RuntimeOptions,
): Promise<Needle2Runtime> {
	return Needle2Runtime.create(options);
}

function resolveGenerateOptions(
	options: Needle2GenerateOptions,
): Needle2ResolvedGenerateOptions {
	const initialization = options.initialization;
	const systemPrompt = defaultSystemPrompt(initialization.systemValues);
	return {
		...options,
		initialization: {
			systemPrompt,
			tools: initialization.tools,
			...(initialization.toolIndexPath === undefined
				? {}
				: { toolIndexPath: initialization.toolIndexPath }),
		},
	};
}

function parseGenerationResult(
	json: string,
	tokenCount: number,
): Needle2GenerationResult {
	let response: Record<string, unknown>;
	try {
		response = JSON.parse(json) as Record<string, unknown>;
	} catch (cause) {
		throw new Needle2ProtocolError("Needle 2 returned malformed JSON.", {
			cause,
		});
	}
	if (
		typeof response.type !== "string" ||
		typeof response.success !== "boolean"
	) {
		throw new Needle2ProtocolError("Needle 2 returned an invalid response.");
	}

	const functionCalls = Array.isArray(response.function_calls)
		? response.function_calls.map((value) => {
				const call = value as { name?: unknown; arguments?: unknown };
				return {
					name: String(call.name ?? ""),
					arguments: (call.arguments ?? {}) as Needle2JSONObject,
				};
			})
		: [];
	const common = {
		type: response.type as Needle2ResponseType,
		functionCalls,
		tokenCount,
		metrics: {
			...property("prefillTokensPerSecond", response.prefill_tps, isNumber),
			...property("decodeTokensPerSecond", response.decode_tps, isNumber),
			...property("peakRAMMegabytes", response.peak_ram_mb, isPositiveNumber),
		},
		...property("reasoning", response.reasoning, isString),
		...property("confidence", response.confidence, isNumber),
	};
	if (response.success) return { success: true, ...common };
	return {
		success: false,
		...common,
		error:
			typeof response.error === "string"
				? response.error
				: "Needle 2 generation failed.",
		...property("errorCode", response.error_code, isString),
	};
}

function property<Key extends string, Value>(
	key: Key,
	value: unknown,
	predicate: (value: unknown) => value is Value,
): { [Property in Key]?: Value } {
	return predicate(value) ? ({ [key]: value } as never) : {};
}

function isString(value: unknown): value is string {
	return typeof value === "string";
}

function isNumber(value: unknown): value is number {
	return typeof value === "number";
}

function isPositiveNumber(value: unknown): value is number {
	return isNumber(value) && value > 0;
}

function requireNativeFactory(): void {
	if (defaultNativeFactory()) {
		return;
	}
	throw new Needle2Error(
		"native-unavailable",
		"The native Needle 2 engine is unavailable in this runtime.",
	);
}
