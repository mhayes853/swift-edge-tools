import { Needle2WorkerBackend } from "./backend.js";
import { createNeedle2Binding } from "./bindings.js";
import {
	defaultAssetURL,
	Needle2Error,
	Needle2ProtocolError,
	PromiseQueue,
} from "./internal.js";
import { formatSystemPrompt, type Needle2SystemValues } from "./system.js";
import type { Needle2JSONObject, Needle2JSONValue } from "./value.js";
import type { Needle2Backend } from "./backend.js";
import type {
	Needle2BinarySource,
	Needle2Factory,
	Needle2WorkerOptions,
} from "./internal.js";

export type Needle2Provider = "direct" | "worker";
export type Needle2Engine = "wasm" | "native" | "auto";

const directRuntimeOperations = new PromiseQueue();

export type Needle2RuntimeOptions<
	Tools extends readonly Needle2AnyTool[] = readonly Needle2ToolDefinition[],
> = {
	tools?: Tools;
	systemValues?: Needle2SystemValues;
	toolIndexPath?: string;
} & (
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
	  }
);

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

export type Needle2ResolvedInitialization = {
	systemPrompt: string;
	tools: readonly Needle2ToolDefinition[];
	toolIndexPath?: string;
};

export type Needle2GenerateOptions = {
	prompt: string;
	maxTokens?: number;
};

export type Needle2ResolvedGenerateOptions = {
	prompt: string;
	initialization: Needle2ResolvedInitialization;
	maxTokens?: number;
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

export type Needle2NonInvokingGeneration<
	Tools extends readonly Needle2AnyTool[] = readonly Needle2ToolDefinition[],
> =
	| (Omit<Needle2GenerationSuccess<Tools>, "functionCalls"> & {
			functionCalls: Needle2UninvokedToolCall<Tools[number]>[];
		})
	| Needle2GenerationFailure<Tools>;

export type Needle2LoopStep<
	Tools extends readonly Needle2AnyTool[] = readonly Needle2ToolDefinition[],
> = {
	generation: Needle2NonInvokingGeneration<Tools>;
	toolResponses: Needle2JSONValue[];
};

export type Needle2LoopTerminationCause =
	| "responded"
	| "refused"
	| "no-tool-calls"
	| "maximum-turns-reached"
	| "failed";

export type Needle2LoopResponse<
	Tools extends readonly Needle2AnyTool[] = readonly Needle2ToolDefinition[],
> = {
	response: Needle2NonInvokingGeneration<Tools>;
	results: Needle2JSONValue[];
	steps: Needle2LoopStep<Tools>[];
	terminationCause: Needle2LoopTerminationCause;
};

export type Needle2LoopTurnParameters = {
	maxTokens?: number;
};

export type Needle2LoopOptions = {
	prompt: string;
	maxTurns?: number;
	maxTokens?: number;
	parameters?: (turn: number) => Needle2LoopTurnParameters;
};

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

export class Needle2Runtime<
	Tools extends readonly Needle2AnyTool[] = readonly Needle2ToolDefinition[],
> {
	readonly provider: Needle2Provider;

	private constructor(
		private readonly backend: Needle2Backend,
		private readonly initialization: Needle2ResolvedInitialization,
		private readonly tools: Tools,
	) {
		this.provider = backend.provider;
	}

	static async create<const Tools extends readonly Needle2AnyTool[]>(
		options: Needle2RuntimeOptions<Tools>,
	): Promise<Needle2Runtime<Tools>> {
		const tools = options.tools ?? ([] as unknown as Tools);
		requireUniqueToolNames(tools);
		const initialization = resolvedInitialization(options, tools);
		const weights = options.weights ?? defaultAssetURL("needle2.cact");
		const engine = options.engine ?? "wasm";
		if (options.provider === "direct") {
			const wasm = options.wasm ?? defaultAssetURL("needle.wasm");
			return new Needle2Runtime(
				await directRuntimeOperations.enqueue(() =>
					createNeedle2Binding(
						engine,
						wasm,
						weights,
						options.factory,
					),
				),
				initialization,
				tools,
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
			initialization,
			tools,
		);
	}

	async generate(
		options: Needle2GenerateOptions,
	): Promise<Needle2GenerationResult<Tools>> {
		const result = await this.perform(() =>
			this.generateNonInvokingInternal(options),
		);
		return (await attachToolOutputs(
			result,
			this.tools,
		)) as Needle2GenerationResult<Tools>;
	}

	/**
	 * Generates one turn without invoking any tool handlers, leaving the calls for the
	 * caller to execute and feed back as the next prompt.
	 */
	async generateNonInvoking(
		options: Needle2GenerateOptions,
	): Promise<Needle2NonInvokingGeneration<Tools>> {
		return this.perform(() => this.generateNonInvokingInternal(options));
	}

	/**
	 * Drives the tool loop: each turn's tool responses become the next turn's prompt.
	 *
	 * Unlike `generate`, a handler that throws does not end the loop. Its error is fed
	 * back to the model as that call's response, so the model can recover on the next
	 * turn, and it stays visible to the caller in the step's `toolResponses`.
	 */
	async runLoop(
		options: Needle2LoopOptions,
	): Promise<Needle2LoopResponse<Tools>> {
		return this.perform(() => this.runLoopInternal(options));
	}

	load(weights: Needle2BinarySource): Promise<void> {
		return this.perform(() => this.backend.load(weights));
	}

	reset(): Promise<void> {
		return this.perform(() => this.backend.reset());
	}

	dispose(): Promise<void> {
		return this.perform(() => this.backend.dispose());
	}

	private async generateNonInvokingInternal(
		options: Needle2GenerateOptions,
	): Promise<Needle2NonInvokingGeneration<Tools>> {
		const generation = await this.backend.generate({
			...options,
			initialization: this.initialization,
		});
		return parseGenerationResult(
			generation.json,
			generation.tokenCount,
		) as Needle2NonInvokingGeneration<Tools>;
	}

	private async runLoopInternal(
		options: Needle2LoopOptions,
	): Promise<Needle2LoopResponse<Tools>> {
		const maxTurns = options.maxTurns ?? 8;
		if (
			!Number.isInteger(maxTurns) ||
			maxTurns < 1 ||
			maxTurns > 8
		) {
			throw new Needle2Error(
				"invalid-turns",
				`Needle 2 supports between 1 and 8 loop turns, but received ${maxTurns}.`,
			);
		}

		const steps: Needle2LoopStep[] = [];
		const results: Needle2JSONValue[] = [];
		let prompt = options.prompt;

		for (let turn = 0; turn < maxTurns; turn += 1) {
			const generation = await this.generateNonInvokingInternal({
				prompt,
				...(options.maxTokens === undefined
					? {}
					: { maxTokens: options.maxTokens }),
				...options.parameters?.(turn),
			});
			if (
				!generation.success ||
				generation.type !== "call" ||
				generation.functionCalls.length === 0
			) {
				steps.push({ generation, toolResponses: [] });
				return {
					response: generation,
					results,
					steps,
					terminationCause: loopTerminationCause(generation),
				} as Needle2LoopResponse<Tools>;
			}

			const outcomes = await toolOutcomes(generation.functionCalls, this.tools);
			const toolResponses = outcomes.map((outcome, index) =>
				toolResponse(outcome, generation.functionCalls[index]?.name ?? ""),
			);
			steps.push({ generation, toolResponses });
			results.push(...toolResponses);
			prompt = JSON.stringify(toolResponses);
		}

		const response = steps[steps.length - 1]?.generation;
		if (!response) {
			throw new Needle2ProtocolError("Needle 2 completed an empty tool loop.");
		}
		return {
			response,
			results,
			steps,
			terminationCause: "maximum-turns-reached",
		} as Needle2LoopResponse<Tools>;
	}

	private perform<Result>(operation: () => Promise<Result>): Promise<Result> {
		return this.provider === "direct"
			? directRuntimeOperations.enqueue(operation)
			: operation();
	}
}

export function needle2<const Tools extends readonly Needle2AnyTool[]>(
	options: Needle2RuntimeOptions<Tools>,
): Promise<Needle2Runtime<Tools>> {
	return Needle2Runtime.create(options);
}

function resolvedInitialization(
	options: Needle2RuntimeOptions<readonly Needle2AnyTool[]>,
	tools: readonly Needle2AnyTool[],
): Needle2ResolvedInitialization {
	return {
		systemPrompt: formatSystemPrompt(options.systemValues),
		tools: tools.map(toolDefinition),
		...(options.toolIndexPath === undefined
			? {}
			: { toolIndexPath: options.toolIndexPath }),
	};
}

function requireUniqueToolNames(tools: readonly Needle2AnyTool[]): void {
	const names = new Set<string>();
	const duplicate = tools.find((tool) => {
		if (names.has(tool.name)) {
			return true;
		}
		names.add(tool.name);
		return false;
	});
	if (duplicate) {
		throw new Needle2Error(
			"duplicate-tool-name",
			`Needle 2 tools must have unique names, but '${duplicate.name}' is declared more than once.`,
		);
	}
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

type ToolOutcome =
	| { status: "fulfilled"; value: unknown }
	| { status: "rejected"; reason: unknown }
	| { status: "unhandled" };

async function toolOutcomes(
	calls: readonly Needle2FunctionCall[],
	tools: readonly Needle2AnyTool[],
): Promise<ToolOutcome[]> {
	const handlers = new Map(
		tools.flatMap((tool) => (tool.call ? [[tool.name, tool.call] as const] : [])),
	);
	const settled = await Promise.allSettled(
		calls.map(async (call) => handlers.get(call.name)?.(call.arguments)),
	);
	return settled.map((outcome, index) => {
		if (outcome.status === "rejected") {
			return { status: "rejected", reason: outcome.reason };
		}
		return handlers.has(calls[index]?.name ?? "")
			? { status: "fulfilled", value: outcome.value }
			: { status: "unhandled" };
	});
}

async function attachToolOutputs(
	result: Needle2GenerationResult,
	tools: readonly Needle2AnyTool[],
): Promise<Needle2GenerationResult> {
	if (
		!result.success ||
		result.functionCalls.length === 0 ||
		tools.every((tool) => !tool.call)
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

	const outcomes = await toolOutcomes(result.functionCalls, tools);
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
		return outcome?.status === "fulfilled"
			? { ...call, output: outcome.value }
			: call;
	}) as Needle2GenerationSuccess["functionCalls"];
	return { ...result, functionCalls };
}

function toolResponse(outcome: ToolOutcome, name: string): Needle2JSONValue {
	if (outcome.status === "fulfilled") {
		return (outcome.value ?? null) as Needle2JSONValue;
	}
	if (outcome.status === "rejected") {
		return {
			error:
				outcome.reason instanceof Error
					? outcome.reason.message
					: String(outcome.reason),
		};
	}
	return { error: `unknown tool: ${name}` };
}

function loopTerminationCause(
	generation: Needle2GenerationResult,
): Needle2LoopTerminationCause {
	if (!generation.success) {
		return "failed";
	}
	if (generation.type === "respond") {
		return "responded";
	}
	if (generation.type === "refuse") {
		return "refused";
	}
	return "no-tool-calls";
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
