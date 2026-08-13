export type Needle2JSONPrimitive = string | number | boolean | null;

export type Needle2JSONValue =
	| Needle2JSONPrimitive
	| Needle2JSONValue[]
	| Needle2JSONObject;

export type Needle2JSONObject = {
	[key: string]: Needle2JSONValue;
};

export type Needle2Provider = "direct" | "worker";

export type Needle2SystemFactValue = string | number | boolean;

export type Needle2SystemValues = {
	[key: string]: Needle2SystemFactValue | undefined;
	date?: string;
	locale?: string;
	device?: string;
	battery?: string;
	network?: string;
	location?: string;
	user?: string;
	assistant?: string;
};

export type Needle2SystemFactProvider = () =>
	| Needle2SystemFactValue
	| null
	| undefined
	| PromiseLike<Needle2SystemFactValue | null | undefined>;

export type Needle2SystemFactOverrides = Readonly<
	Record<string, Needle2SystemFactValue | null>
>;

export type Needle2SystemFactProviders = Readonly<
	Record<string, Needle2SystemFactProvider>
>;

export type Needle2SystemFactsProvider = () =>
	| Needle2SystemValues
	| PromiseLike<Needle2SystemValues>;

export type Needle2SystemValuesOptions = {
	overrides?: Needle2SystemFactOverrides;
	providers?: Needle2SystemFactProviders;
	includeLocation?: boolean;
};

export type Needle2BinarySource = string | URL | ArrayBuffer | Uint8Array;

export type Needle2Factory = (options: {
	wasmBinary: Uint8Array;
}) => unknown | PromiseLike<unknown>;

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

export type Needle2NativeGeneration = {
	json: string;
	tokenCount: number;
};

export interface Needle2Backend {
	readonly provider: Needle2Provider;

	generate(
		options: Needle2ResolvedGenerateOptions,
	): Promise<Needle2NativeGeneration>;
	load(weights: Needle2BinarySource): Promise<void>;
	dispose(): Promise<void>;
}
