import type { Needle2NativeGeneration } from "./bindings.js";
import {
	defaultAssetURL,
	Needle2Error,
	isNodeLikeEnvironment,
	serializeBinarySource,
} from "./internal.js";
import type {
	Needle2BinarySource,
	Needle2SerializedBinarySource,
	Needle2WorkerOptions,
} from "./internal.js";
import type {
	Needle2Engine,
	Needle2Provider,
	Needle2ResolvedGenerateOptions,
} from "./runtime.js";

export { Needle2WASMBinding as Needle2DirectBackend } from "./bindings.js";
export type { Needle2NativeGeneration } from "./bindings.js";

export interface Needle2Backend {
	readonly provider: Needle2Provider;
	generate(
		options: Needle2ResolvedGenerateOptions,
	): Promise<Needle2NativeGeneration>;
	load(weights: Needle2BinarySource): Promise<void>;
	reset(): Promise<void>;
	dispose(): Promise<void>;
}

export type Needle2WorkerRequest =
	| {
			id: number;
			operation: "initialize";
			wasm: Needle2SerializedBinarySource;
			weights: Needle2SerializedBinarySource;
			engine: Needle2Engine;
	  }
	| {
			id: number;
			operation: "generate";
			options: Needle2ResolvedGenerateOptions;
	  }
	| { id: number; operation: "load"; weights: Needle2SerializedBinarySource }
	| { id: number; operation: "reset" }
	| { id: number; operation: "dispose" };

export type Needle2WorkerResult = Needle2NativeGeneration | undefined;

export type Needle2SerializedError = {
	name: string;
	message: string;
	stack?: string;
	code?: string;
};

export type Needle2WorkerResponse =
	| { id: number; success: true; result?: Needle2WorkerResult }
	| { id: number; success: false; error: Needle2SerializedError };

type WithoutID<Request> = Request extends { id: number }
	? Omit<Request, "id">
	: never;
type Needle2WorkerRequestInput = WithoutID<Needle2WorkerRequest>;

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

	private readonly pending = new Map<number, PendingRequest>();
	private nextRequestID = 0;
	private terminalError: Error | undefined;
	private disposePromise: Promise<void> | undefined;

	static async create(
		wasm: Needle2BinarySource,
		weights: Needle2BinarySource,
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

	async generate(
		options: Needle2ResolvedGenerateOptions,
	): Promise<Needle2NativeGeneration> {
		const result = await this.request({ operation: "generate", options });
		if (!result) {
			throw new Needle2Error(
				"worker-protocol",
				"The Needle 2 worker returned no generation.",
			);
		}
		return result;
	}

	async load(weights: Needle2BinarySource): Promise<void> {
		await this.request({
			operation: "load",
			weights: serializeBinarySource(weights),
		});
	}

	async reset(): Promise<void> {
		await this.request({ operation: "reset" });
	}

	dispose(): Promise<void> {
		this.disposePromise ??= this.disposeInternal();
		return this.disposePromise;
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

	private request(
		request: Needle2WorkerRequestInput,
	): Promise<Needle2WorkerResult> {
		if (this.terminalError) {
			return Promise.reject(this.terminalError);
		}
		const id = this.nextRequestID++;
		return new Promise((resolve, reject) => {
			this.pending.set(id, { resolve, reject });
			try {
				this.connection.postMessage({ ...request, id } as Needle2WorkerRequest);
			} catch (error) {
				this.pending.delete(id);
				reject(error);
			}
		});
	}

	private receive(response: Needle2WorkerResponse): void {
		const pending = this.pending.get(response.id);
		if (!pending) return;

		this.pending.delete(response.id);
		if (response.success) {
			pending.resolve(response.result);
		} else {
			pending.reject(deserializeError(response.error));
		}
	}

	private failPending(error: Error): void {
		for (const pending of this.pending.values()) pending.reject(error);
		this.pending.clear();
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
