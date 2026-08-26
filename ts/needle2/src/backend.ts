import { nativeBinding, wasmBinding } from "./bindings.js";
import type {
	Needle2Binding,
	Needle2NativeGeneration,
} from "./bindings.js";
import {
	binarySourceBytes,
	defaultAssetURL,
	Needle2Error,
	isNodeLikeEnvironment,
	PromiseQueue,
	serializeBinarySource,
} from "./internal.js";
import type {
	Needle2BinarySource,
	Needle2Factory,
	Needle2SerializedBinarySource,
	Needle2WorkerOptions,
} from "./internal.js";
import type {
	Needle2Engine,
	Needle2CompletionOptions,
	Needle2Provider,
	Needle2ResolvedInitialization,
} from "./runtime.js";

export type { Needle2NativeGeneration } from "./bindings.js";

export type Needle2Complete = (
	options: Needle2CompletionOptions,
) => Promise<Needle2NativeGeneration>;

export interface Needle2Backend {
	readonly provider: Needle2Provider;
	withModel<Result>(
		operation: (complete: Needle2Complete) => Promise<Result>,
	): Promise<Result>;
	load(weights: Needle2BinarySource): Promise<void>;
	reset(): Promise<void>;
	dispose(): Promise<void>;
}

type WeightIdentity = object | string;

class DirectModel {
	readonly operations = new PromiseQueue();
	activeBackend: Needle2DirectBackend | undefined;
	loadedWeightsIdentity: WeightIdentity | undefined;

	constructor(readonly binding: Needle2Binding) {}
}

let sharedDirectNativeModel: Promise<DirectModel> | undefined;
let sharedDirectWasmModel: Promise<DirectModel> | undefined;
let sharedDirectWasmFactory: Needle2Factory | undefined;

export class Needle2DirectBackend implements Needle2Backend {
	readonly provider = "direct" as const;

	private disposed = false;
	private weightsIdentity: WeightIdentity;

	private constructor(
		private readonly model: DirectModel,
		private readonly initialization: Needle2ResolvedInitialization,
		private weights: Uint8Array,
		weightsIdentity: WeightIdentity,
	) {
		this.weightsIdentity = weightsIdentity;
	}

	static async create(
		engine: Needle2Engine,
		wasm: Needle2BinarySource,
		weights: Needle2BinarySource,
		initialization: Needle2ResolvedInitialization,
		factory?: Needle2Factory,
	): Promise<Needle2DirectBackend> {
		const weightBytes = await binarySourceBytes(weights);
		const model = await directModel(engine, wasm, factory);
		const backend = new Needle2DirectBackend(
			model,
			initialization,
			weightBytes,
			binarySourceIdentity(weights),
		);
		return model.operations.enqueue(() => {
			backend.loadWeights();
			return backend;
		});
	}

	withModel<Result>(
		operation: (complete: Needle2Complete) => Promise<Result>,
	): Promise<Result> {
		return this.model.operations.enqueue(() => {
			this.requireActive();
			this.activate();
			return operation(async (options) => this.model.binding.complete(options));
		});
	}

	async load(weights: Needle2BinarySource): Promise<void> {
		const loadedWeights = await binarySourceBytes(weights);
		return this.model.operations.enqueue(() => {
			this.requireActive();
			if (this.model.activeBackend === this) {
				throw new Needle2Error(
					"active-session",
					"Reset the active Needle 2 runtime before loading new weights.",
				);
			}
			const weightsIdentity = {};
			this.loadWeights(loadedWeights, weightsIdentity);
			this.weights = loadedWeights;
			this.weightsIdentity = weightsIdentity;
		});
	}

	reset(): Promise<void> {
		return this.model.operations.enqueue(() => {
			this.requireActive();
			this.deactivate();
		});
	}

	dispose(): Promise<void> {
		return this.model.operations.enqueue(() => {
			if (this.disposed) {
				return;
			}
			this.deactivate();
			this.disposed = true;
		});
	}

	private activate(): void {
		this.loadWeights();
		if (this.model.activeBackend !== this) {
			this.model.binding.initialize(this.initialization);
			this.model.activeBackend = this;
		}
	}

	private loadWeights(
		weights = this.weights,
		weightsIdentity = this.weightsIdentity,
	): void {
		if (this.model.loadedWeightsIdentity === weightsIdentity) {
			return;
		}
		this.model.binding.load(weights);
		this.model.loadedWeightsIdentity = weightsIdentity;
		this.model.activeBackend = undefined;
	}

	private deactivate(): void {
		if (this.model.activeBackend === this) {
			this.model.binding.reset();
			this.model.activeBackend = undefined;
		}
	}

	private requireActive(): void {
		if (this.disposed) {
			throw new Needle2Error(
				"disposed",
				"This Needle 2 runtime has been disposed.",
			);
		}
	}
}

function directModel(
	engine: Needle2Engine,
	wasm: Needle2BinarySource,
	factory?: Needle2Factory,
): Promise<DirectModel> {
	if (engine === "native") {
		return directNativeModel();
	}
	if (engine === "auto") {
		return directNativeModel().catch(() => directWasmModel(wasm, factory));
	}
	return directWasmModel(wasm, factory);
}

function directNativeModel(): Promise<DirectModel> {
	if (sharedDirectNativeModel) {
		return sharedDirectNativeModel;
	}
	const modelPromise = nativeBinding().then((binding) => new DirectModel(binding));
	const cachedPromise = modelPromise.catch((error) => {
		if (sharedDirectNativeModel === cachedPromise) {
			sharedDirectNativeModel = undefined;
		}
		throw error;
	});
	sharedDirectNativeModel = cachedPromise;
	return cachedPromise;
}

function directWasmModel(
	wasm: Needle2BinarySource,
	factory?: Needle2Factory,
): Promise<DirectModel> {
	if (sharedDirectWasmModel && sharedDirectWasmFactory === factory) {
		return sharedDirectWasmModel;
	}
	sharedDirectWasmFactory = factory;
	const modelPromise = binarySourceBytes(wasm).then(async (bytes) =>
		new DirectModel(await wasmBinding(bytes, factory)),
	);
	const cachedPromise = modelPromise.catch((error) => {
		if (sharedDirectWasmModel === cachedPromise) {
			sharedDirectWasmModel = undefined;
			sharedDirectWasmFactory = undefined;
		}
		throw error;
	});
	sharedDirectWasmModel = cachedPromise;
	return cachedPromise;
}

export type Needle2WorkerRequest =
	| {
			operation: "initialize";
			wasm: Needle2SerializedBinarySource;
			weights: Needle2SerializedBinarySource;
			engine: Needle2Engine;
			initialization: Needle2ResolvedInitialization;
	  }
	| {
			operation: "generate";
			options: Needle2CompletionOptions;
	  }
	| { operation: "load"; weights: Needle2SerializedBinarySource }
	| { operation: "reset" }
	| { operation: "dispose" };

export type Needle2WorkerResult = Needle2NativeGeneration | undefined;

export type Needle2SerializedError = {
	name: string;
	message: string;
	stack?: string;
	code?: string;
};

export type Needle2WorkerResponse =
	| { success: true; result?: Needle2WorkerResult }
	| { success: false; error: Needle2SerializedError };

type WorkerConnection = {
	postMessage(message: Needle2WorkerRequest): void;
	onMessage(handler: (message: Needle2WorkerResponse) => void): void;
	onFailure(handler: (error: Error) => void): void;
	terminate(): Promise<void>;
};

type PendingRequest = {
	resolve(value: Needle2WorkerResult): void;
	reject(reason: unknown): void;
};

export class Needle2WorkerBackend implements Needle2Backend {
	readonly provider = "worker" as const;

	private readonly operations = new PromiseQueue();
	private pending: PendingRequest | undefined;
	private terminalError: Error | undefined;
	private disposePromise: Promise<void> | undefined;

	static async create(
		wasm: Needle2BinarySource,
		weights: Needle2BinarySource,
		initialization: Needle2ResolvedInitialization,
		options?: Needle2WorkerOptions,
		engine: Needle2Engine = "wasm",
	): Promise<Needle2WorkerBackend> {
		const connection = await workerConnection(
			defaultAssetURL("needle2.worker.mjs"),
			options,
		);
		const backend = new Needle2WorkerBackend(connection);
		try {
			await backend.request({
				operation: "initialize",
				wasm: serializeBinarySource(wasm),
				weights: serializeBinarySource(weights),
				engine,
				initialization,
			});
			return backend;
		} catch (error) {
			await connection.terminate();
			throw error;
		}
	}

	private constructor(private readonly connection: WorkerConnection) {
		connection.onMessage((message) => this.receive(message));
		connection.onFailure((error) => this.handleFailure(error));
	}

	withModel<Result>(
		operation: (complete: Needle2Complete) => Promise<Result>,
	): Promise<Result> {
		return this.operations.enqueue(() =>
			operation((options) => this.complete(options)),
		);
	}

	load(weights: Needle2BinarySource): Promise<void> {
		return this.operations.enqueue(async () => {
			await this.request({
				operation: "load",
				weights: serializeBinarySource(weights),
			});
		});
	}

	reset(): Promise<void> {
		return this.operations.enqueue(async () => {
			await this.request({ operation: "reset" });
		});
	}

	dispose(): Promise<void> {
		this.disposePromise ??= this.operations.enqueue(() =>
			this.disposeInternal(),
		);
		return this.disposePromise;
	}

	private async complete(
		options: Needle2CompletionOptions,
	): Promise<Needle2NativeGeneration> {
		return this.request({ operation: "generate", options }) as Promise<Needle2NativeGeneration>;
	}

	private async disposeInternal(): Promise<void> {
		if (this.terminalError) {
			await this.connection.terminate();
			return;
		}
		const disposedError = new Needle2Error(
			"disposed",
			"This Needle 2 runtime has been disposed.",
		);
		const request = this.request({ operation: "dispose" });
		this.terminalError = disposedError;
		try {
			await request;
		} finally {
			await this.connection.terminate();
			this.failPending(disposedError);
		}
	}

	private request(request: Needle2WorkerRequest): Promise<Needle2WorkerResult> {
		if (this.terminalError) {
			return Promise.reject(this.terminalError);
		}
		return new Promise((resolve, reject) => {
			this.pending = { resolve, reject };
			try {
				this.connection.postMessage(request);
			} catch (error) {
				this.pending = undefined;
				reject(error);
			}
		});
	}

	private receive(response: Needle2WorkerResponse): void {
		const pending = this.pending;
		if (!pending) {
			return;
		}

		this.pending = undefined;
		if (response.success) {
			pending.resolve(response.result);
		} else {
			pending.reject(deserializeError(response.error));
		}
	}

	private failPending(error: Error): void {
		this.pending?.reject(error);
		this.pending = undefined;
	}

	private handleFailure(error: Error): void {
		if (this.terminalError) {
			this.failPending(this.terminalError);
			return;
		}
		this.terminalError = error;
		this.failPending(error);
		void this.connection.terminate().catch(() => undefined);
	}
}

async function workerConnection(
	url: URL,
	options?: Needle2WorkerOptions,
): Promise<WorkerConnection> {
	if (typeof Worker !== "undefined") {
		return browserWorkerConnection(url, options);
	}
	if (!isNodeLikeEnvironment()) {
		throw new Needle2Error(
			"worker-unavailable",
			"This runtime does not provide Web Workers or Node worker threads.",
		);
	}

	const workerThreadsSpecifier = "node:worker_threads";
	const { Worker: NodeWorker } = await import(
		/* @vite-ignore */ workerThreadsSpecifier
	);
	const worker = new NodeWorker(
		url,
		options?.name ? { name: options.name } : undefined,
	);
	return {
		postMessage: (message) => worker.postMessage(message),
		onMessage: (handler) => worker.on("message", handler),
		onFailure(handler) {
			worker.on("error", handler);
			worker.on("exit", (code: number) =>
				handler(
					new Needle2Error(
						"worker-exited",
						`The Needle 2 worker exited with status ${code}.`,
					),
				),
			);
		},
		async terminate() {
			await worker.terminate();
		},
	};
}

function browserWorkerConnection(
	url: URL,
	options?: Needle2WorkerOptions,
): WorkerConnection {
	const worker = new Worker(url, { ...options, type: "module" });
	return {
		postMessage: (message) => worker.postMessage(message),
		onMessage: (handler) =>
			worker.addEventListener("message", (event) =>
				handler(event.data as Needle2WorkerResponse),
			),
		onFailure(handler) {
			worker.addEventListener("error", (event) =>
				handler(new Error(event.message)),
			);
			worker.addEventListener("messageerror", () =>
				handler(
					new Needle2Error(
						"worker-message-error",
						"The Needle 2 worker received an unreadable message.",
					),
				),
			);
		},
		async terminate() {
			worker.terminate();
		},
	};
}

function deserializeError(error: Needle2SerializedError): Error {
	const result = new Needle2Error(error.code ?? "worker-error", error.message);
	result.name = error.name;
	if (error.stack) result.stack = error.stack;
	return result;
}

function binarySourceIdentity(source: Needle2BinarySource): WeightIdentity {
	if (typeof source === "string") {
		return source;
	}
	if (source instanceof URL) {
		return source.href;
	}
	return source;
}
