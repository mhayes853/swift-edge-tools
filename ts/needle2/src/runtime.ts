import { Needle2DirectBackend, Needle2WorkerBackend } from "./backend.js";
import {
	defaultAssetURL,
	Needle2Error,
} from "./internal.js";
import { formatSystemPrompt, type Needle2SystemValues } from "./system.js";
import type { Needle2JSONObject, Needle2JSONValue } from "./value.js";
import type { Needle2Backend, Needle2Complete } from "./backend.js";
import type {
	Needle2BinarySource,
	Needle2Factory,
	Needle2WorkerOptions,
} from "./internal.js";

export type Needle2Provider = "direct" | "worker";
export type Needle2Engine = "wasm" | "native" | "auto";

const NEEDLE_LOOP_MAX_TURNS = 8

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

export type Needle2FunctionCall<Tool = Needle2ToolDefinition> = Tool extends {
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

export type Needle2CompletionOptions = {
	prompt: string;
	maxTokens: number;
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
	functionCalls: Needle2FunctionCall<Tools[number]>[];
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
	functionCalls: Needle2FunctionCall<Tools[number]>[];
	reasoning?: string;
	confidence?: number;
	tokenCount: number;
	metrics: Needle2GenerationMetrics;
};

export type Needle2GenerationResult<
	Tools extends readonly Needle2AnyTool[] = readonly Needle2ToolDefinition[],
> = Needle2GenerationSuccess<Tools> | Needle2GenerationFailure<Tools>;

export type Needle2LoopStep<
	Tools extends readonly Needle2AnyTool[] = readonly Needle2ToolDefinition[],
> = {
	generation: Needle2GenerationResult<Tools>;
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
	response: Needle2GenerationResult<Tools>;
	steps: Needle2LoopStep<Tools>[];
	terminationCause: Needle2LoopTerminationCause;
};

export type Needle2LoopOptions = {
	prompt: string;
	maxTurns?: number;
	maxTokens?: number;
};

export class Needle2Runtime<
	Tools extends readonly Needle2AnyTool[] = readonly Needle2ToolDefinition[],
> {
	readonly provider: Needle2Provider;

	private constructor(
		private readonly backend: Needle2Backend,
		private readonly tools: ReadonlyMap<string, Needle2AnyTool>,
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
				await Needle2DirectBackend.create(
					engine,
					wasm,
					weights,
					initialization,
					options.factory,
				),
				toolMap(tools),
			);
		}

		const wasm = options.wasm ?? defaultAssetURL("needle.wasm");
		return new Needle2Runtime(
			await Needle2WorkerBackend.create(
				wasm,
				weights,
				initialization,
				options.workerOptions,
				engine,
			),
			toolMap(tools),
		);
	}

	async generate(
		options: Needle2GenerateOptions,
	): Promise<Needle2GenerationResult<Tools>> {
		return this.backend.withModel((complete) =>
			this.generateInternal(complete, options),
		);
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
		return this.backend.withModel((complete) =>
			this._runLoop(complete, options),
		);
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

	private async generateInternal(
		complete: Needle2Complete,
		options: Needle2GenerateOptions,
	): Promise<Needle2GenerationResult<Tools>> {
		const generation = await complete(completionOptions(options));
		return parseGenerationResult(
			generation.json,
			generation.tokenCount,
		) as Needle2GenerationResult<Tools>;
	}

	private async _runLoop(
		complete: Needle2Complete,
		options: Needle2LoopOptions,
	): Promise<Needle2LoopResponse<Tools>> {
		const maxTurns = options.maxTurns ?? NEEDLE_LOOP_MAX_TURNS;
		validateLoopMaxTurns(maxTurns)

		const steps: Needle2LoopStep[] = [];
		let prompt = options.prompt;

		for (let turn = 0; turn < maxTurns; turn += 1) {
			const generation = await this.generateInternal(complete, {
				prompt,
				...(options.maxTokens === undefined
					? {}
					: { maxTokens: options.maxTokens }),
			});
			if (
				!generation.success ||
				generation.type !== "call" ||
				generation.functionCalls.length === 0
			) {
				steps.push({ generation, toolResponses: [] });
				return {
					response: generation,
					steps,
					terminationCause: loopTerminationCause(generation),
				} as Needle2LoopResponse<Tools>;
			}

			const outcomes = await toolOutcomes(generation.functionCalls, this.tools);
			const toolResponses = outcomes.map((outcome, index) =>
				toolResponse(outcome, generation.functionCalls[index]?.name ?? ""),
			);
			steps.push({ generation, toolResponses });
			prompt = JSON.stringify(toolResponses);
		}

		return {
			response: steps[steps.length - 1]!.generation,
			steps,
			terminationCause: "maximum-turns-reached",
		} as Needle2LoopResponse<Tools>;
	}

}

export function needle2<const Tools extends readonly Needle2AnyTool[]>(
	options: Needle2RuntimeOptions<Tools>,
): Promise<Needle2Runtime<Tools>> {
	return Needle2Runtime.create(options);
}

function validateLoopMaxTurns(turns: number) {
  if (turns < 1 || turns > NEEDLE_LOOP_MAX_TURNS) {
		throw new Needle2Error(
			"invalid-turns",
			`Needle 2 supports between 1 and 8 loop turns, but received ${turns}.`,
		);
	}
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

function toolMap(
	tools: readonly Needle2AnyTool[],
): ReadonlyMap<string, Needle2AnyTool> {
	return new Map(tools.map((tool) => [tool.name, tool]));
}

type ToolOutcome =
	| { status: "fulfilled"; value: unknown }
	| { status: "rejected"; reason: unknown }
	| { status: "unhandled" };

async function toolOutcomes(
	calls: readonly Needle2FunctionCall[],
	tools: ReadonlyMap<string, Needle2AnyTool>,
): Promise<ToolOutcome[]> {
	const settled = await Promise.allSettled(
		calls.map(async (call) => tools.get(call.name)?.call?.(call.arguments)),
	);
	return settled.map((outcome, index) => {
		if (outcome.status === "rejected") {
			return { status: "rejected", reason: outcome.reason };
		}
		return tools.get(calls[index]!.name)?.call
			? { status: "fulfilled", value: outcome.value }
			: { status: "unhandled" };
	});
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
	const response = JSON.parse(json) as {
		success: boolean;
		type: Needle2ResponseType;
		function_calls?: Needle2FunctionCall[];
		error?: string;
		error_code?: string;
		reasoning?: string;
		confidence?: number;
		prefill_tps?: number;
		decode_tps?: number;
		peak_ram_mb?: number;
	};
	const common = {
		type: response.type,
		functionCalls: response.function_calls ?? [],
		tokenCount,
		metrics: {
			...optional("prefillTokensPerSecond", response.prefill_tps),
			...optional("decodeTokensPerSecond", response.decode_tps),
			...optional("peakRAMMegabytes", response.peak_ram_mb || undefined),
		},
		...optional("reasoning", response.reasoning),
		...optional("confidence", response.confidence),
	};
	if (response.success) return { success: true, ...common };
	return {
		success: false,
		...common,
		error: response.error ?? "Needle 2 generation failed.",
		...optional("errorCode", response.error_code),
	};
}

function optional<Key extends string, Value>(
	key: Key,
	value: Value | undefined,
): { [Property in Key]?: Value } {
	return value === undefined ? {} : ({ [key]: value } as never);
}

function completionOptions(
	options: Needle2GenerateOptions,
): Needle2CompletionOptions {
	return {
		prompt: options.prompt,
		maxTokens: positiveInteger(options.maxTokens ?? 256, "maxTokens"),
	};
}

function positiveInteger(value: number, name: string): number {
	if (!Number.isInteger(value) || value <= 0 || value > 2_147_483_647) {
		throw new Needle2Error(
			"invalid-generate-options",
			`Needle 2 ${name} must be an integer between 1 and 2147483647.`,
		);
	}
	return value;
}
