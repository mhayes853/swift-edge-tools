import { Needle2DirectBackend, Needle2WorkerBackend } from "./backend";
import {
	defaultAssetURL,
	Needle2ProtocolError,
	setAssetBaseURL,
} from "./internal";
import {
	defaultSystemValues,
	defaultSystemPrompt,
	type Needle2SystemFactsProvider,
	type Needle2SystemValuesOptions,
} from "./system";
import type { Needle2JSONObject } from "./value";
import type { Needle2Backend } from "./backend";
import type { Needle2BinarySource } from "./internal";

export type Needle2Provider = "direct" | "worker";

export type Needle2Factory = (options: {
	wasmBinary: Uint8Array;
}) => unknown | PromiseLike<unknown>;

export type Needle2RuntimeOptions =
	| {
			provider: "direct";
			wasm?: Needle2BinarySource;
			weights?: Needle2BinarySource;
			factory?: Needle2Factory;
	  }
	| {
			provider: "worker";
			wasm?: Needle2BinarySource;
			weights?: Needle2BinarySource;
			workerURL?: string | URL;
			workerOptions?: WorkerOptions;
	  };

export type Needle2ToolDefinition = {
	name: string;
	description?: string;
	parameters: Needle2JSONObject;
};

export type Needle2Initialization = {
	systemPrompt?: string;
	systemFacts?: Needle2SystemFactsProvider;
	systemFactsOptions?: Needle2SystemValuesOptions;
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
		const wasm = options.wasm ?? defaultAssetURL("needle.wasm");
		const weights = options.weights ?? defaultAssetURL("needle2.cact");
		if (options.provider === "direct") {
			return new Needle2Runtime(
				await Needle2DirectBackend.create(wasm, weights, options.factory),
			);
		}

		const workerURL = options.workerURL
			? new URL(options.workerURL, defaultAssetURL("./"))
			: defaultAssetURL("needle2.worker.mjs");
		return new Needle2Runtime(
			await Needle2WorkerBackend.create(
				workerURL,
				wasm,
				weights,
				options.workerOptions,
			),
		);
	}

	async generate(
		options: Needle2GenerateOptions,
	): Promise<Needle2GenerationResult> {
		const generation = await this.backend.generate(
			await resolveGenerateOptions(options),
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

async function resolveGenerateOptions(
	options: Needle2GenerateOptions,
): Promise<Needle2ResolvedGenerateOptions> {
	const initialization = options.initialization;
	let systemPrompt = initialization.systemPrompt;
	if (systemPrompt === undefined) {
		const facts = initialization.systemFacts
			? await initialization.systemFacts()
			: await defaultSystemValues(initialization.systemFactsOptions);
		systemPrompt = defaultSystemPrompt(facts);
	}
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
	if (response.success) {
		return { success: true, ...common };
	}
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
