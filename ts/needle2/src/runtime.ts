import { Needle2WorkerBackend } from "./backend.js";
import { createNeedle2Binding } from "./bindings.js";
import {
	defaultAssetURL,
	Needle2Error,
	Needle2ProtocolError,
} from "./internal.js";
import { defaultSystemPrompt, type Needle2SystemValues } from "./system.js";
import type { Needle2JSONObject, Needle2JSONValue } from "./value.js";
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

export type Needle2ToolOutput = Needle2JSONValue | void;

export type Needle2ToolHandler<
	Arguments extends Needle2JSONObject = Needle2JSONObject,
	Output extends Needle2ToolOutput = Needle2ToolOutput,
> = (args: Arguments) => Output | PromiseLike<Output>;

export type Needle2Tool<
	Name extends string = string,
	Arguments extends Needle2JSONObject = Needle2JSONObject,
	Output extends Needle2ToolOutput = Needle2ToolOutput,
> = {
	name: Name;
	description?: string;
	parameters: Needle2JSONObject;
	call: Needle2ToolHandler<Arguments, Output>;
};

export type Needle2AnyTool = Needle2ToolDefinition & {
	call?: (args: any) => unknown;
};

type ToolArguments<Tool> = Tool extends {
	call: (args: infer Arguments) => unknown;
}
	? Arguments
	: Needle2JSONObject;

type ToolOutput<Tool> = Tool extends { call: (args: never) => infer Output }
	? Awaited<Output>
	: never;

export type Needle2InvokedToolCall<Tool> = Tool extends {
	name: infer Name extends string;
}
	? Tool extends { call: unknown }
		? {
				name: Name;
				arguments: ToolArguments<Tool>;
				output: ToolOutput<Tool>;
			}
		: { name: Name; arguments: Needle2JSONObject }
	: never;

export type Needle2UninvokedToolCall<Tool> = Tool extends {
	name: infer Name extends string;
}
	? { name: Name; arguments: ToolArguments<Tool> }
	: never;

export type Needle2Initialization<
	Tools extends readonly Needle2AnyTool[] = readonly Needle2ToolDefinition[],
> = {
	systemValues?: Needle2SystemValues;
	tools: Tools;
	toolIndexPath?: string;
};

export type Needle2ResolvedInitialization = {
	systemPrompt: string;
	tools: readonly Needle2ToolDefinition[];
	toolIndexPath?: string;
};

export type Needle2GenerateOptions<
	Tools extends readonly Needle2AnyTool[] = readonly Needle2ToolDefinition[],
> = {
	prompt: string;
	initialization: Needle2Initialization<Tools>;
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

export type Needle2GenerationSuccess<
	Tools extends readonly Needle2AnyTool[] = readonly Needle2ToolDefinition[],
> = {
	success: true;
	type: Needle2ResponseType;
	functionCalls: Needle2InvokedToolCall<Tools[number]>[];
	reasoning?: string;
	confidence?: number;
	tokenCount: number;
	metrics: Needle2GenerationMetrics;
};

export type Needle2GenerationFailure<
	Tools extends readonly Needle2AnyTool[] = readonly Needle2ToolDefinition[],
> = {
	success: false;
	type: Needle2ResponseType;
	error: string;
	errorCode?: string;
	functionCalls: Needle2UninvokedToolCall<Tools[number]>[];
	reasoning?: string;
	confidence?: number;
	tokenCount: number;
	metrics: Needle2GenerationMetrics;
};

export type Needle2GenerationResult<
	Tools extends readonly Needle2AnyTool[] = readonly Needle2ToolDefinition[],
> = Needle2GenerationSuccess<Tools> | Needle2GenerationFailure<Tools>;

export type Needle2ToolCallFailure = {
	name: string;
	index: number;
	cause: unknown;
};

export class Needle2ToolCallError extends Needle2Error {
	constructor(
		readonly generation: Needle2GenerationSuccess,
		readonly failures: readonly Needle2ToolCallFailure[],
	) {
		super(
			"tool-call-failed",
			`The ${failures.map((failure) => `'${failure.name}'`).join(", ")} tool handlers threw while responding to Needle 2.`,
			{ cause: failures[0]?.cause },
		);
		this.name = "Needle2ToolCallError";
	}
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
			const wasm = options.wasm ?? defaultAssetURL("needle.wasm");
			return new Needle2Runtime(
				await createNeedle2Binding(
					engine,
					wasm,
					weights,
					options.factory,
				),
			);
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

	async generate<const Tools extends readonly Needle2AnyTool[]>(
		options: Needle2GenerateOptions<Tools>,
	): Promise<Needle2GenerationResult<Tools>> {
		const generation = await this.backend.generate(
			resolveGenerateOptions(options),
		);
		const result = parseGenerationResult(
			generation.json,
			generation.tokenCount,
		);
		return (await invokeTools(
			result,
			options.initialization.tools,
		)) as Needle2GenerationResult<Tools>;
	}

	load(weights: Needle2BinarySource): Promise<void> {
		return this.backend.load(weights);
	}

	reset(): Promise<void> {
		return this.backend.reset();
	}

	dispose(): Promise<void> {
		return this.backend.dispose();
	}
}

export function needle2(
	options: Needle2RuntimeOptions,
): Promise<Needle2Runtime> {
	return Needle2Runtime.create(options);
}

function resolveGenerateOptions(
	options: Needle2GenerateOptions<readonly Needle2AnyTool[]>,
): Needle2ResolvedGenerateOptions {
	const initialization = options.initialization;
	const systemPrompt = defaultSystemPrompt(initialization.systemValues);
	return {
		...options,
		initialization: {
			systemPrompt,
			tools: initialization.tools.map(toolDefinition),
			...(initialization.toolIndexPath === undefined
				? {}
				: { toolIndexPath: initialization.toolIndexPath }),
		},
	};
}

function toolDefinition(tool: Needle2AnyTool): Needle2ToolDefinition {
	return {
		name: tool.name,
		...(tool.description === undefined
			? {}
			: { description: tool.description }),
		parameters: tool.parameters,
	};
}

async function invokeTools(
	result: Needle2GenerationResult,
	tools: readonly Needle2AnyTool[],
): Promise<Needle2GenerationResult> {
	const handlers = new Map(
		tools.flatMap((tool) => (tool.call ? [[tool.name, tool.call] as const] : [])),
	);
	if (
		!result.success ||
		handlers.size === 0 ||
		result.functionCalls.length === 0
	) {
		return result;
	}

	const unknownCall = result.functionCalls.find((call) =>
		tools.every((tool) => tool.name !== call.name),
	);
	if (unknownCall) {
		throw new Needle2ProtocolError(
			`Needle 2 called the unknown '${unknownCall.name}' tool.`,
		);
	}

	// The callback is `async` so that a handler throwing synchronously rejects
	// alongside the others instead of escaping `map` and orphaning them.
	const outcomes = await Promise.allSettled(
		result.functionCalls.map(async (call) =>
			handlers.get(call.name)?.(call.arguments),
		),
	);
	const failures = outcomes.flatMap((outcome, index) =>
		outcome.status === "rejected"
			? [
					{
						name: result.functionCalls[index]?.name ?? "",
						index,
						cause: outcome.reason,
					},
				]
			: [],
	);
	if (failures.length > 0) {
		throw new Needle2ToolCallError(result, failures);
	}

	const functionCalls = result.functionCalls.map((call, index) => {
		const outcome = outcomes[index];
		return handlers.has(call.name) && outcome?.status === "fulfilled"
			? { ...call, output: outcome.value }
			: call;
	}) as Needle2GenerationSuccess["functionCalls"];
	return { ...result, functionCalls };
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
