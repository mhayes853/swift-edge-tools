import type { Needle2NativeGeneration } from "./bindings";
import {
	defaultAssetURL,
	Needle2Error,
	isBrowserEnvironment,
	serializeBinarySource,
} from "./internal";
import type {
	Needle2BinarySource,
	Needle2SerializedBinarySource,
} from "./internal";
import type {
	Needle2Provider,
	Needle2ResolvedGenerateOptions,
} from "./runtime";

export { Needle2WASMBinding as Needle2DirectBackend } from "./bindings";
export type { Needle2NativeGeneration } from "./bindings";

export interface Needle2Backend {
	readonly provider: Needle2Provider;
	generate(
		options: Needle2ResolvedGenerateOptions,
	): Promise<Needle2NativeGeneration>;
	load(weights: Needle2BinarySource): Promise<void>;
	dispose(): Promise<void>;
}

export type Needle2WorkerRequest =
	| {
			id: number;
			operation: "initialize";
			wasm: Needle2SerializedBinarySource;
			weights: Needle2SerializedBinarySource;
			engine: "wasm" | "native";
	  }
	| {
			id: number;
			operation: "generate";
			options: Needle2ResolvedGenerateOptions;
	  }
	| { id: number; operation: "load"; weights: Needle2SerializedBinarySource }
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

type Needle2WorkerRequestInput = Needle2WorkerRequest extends infer Request
	? Request extends { id: number }
		? Omit<Request, "id">
		: never
	: never;

type WorkerConnection = {
	postMessage(message: Needle2WorkerRequest): void;
	onMessage(handler: (message: Needle2WorkerResponse) => void): void;
	onError(handler: (error: Error) => void): void;
	terminate(): void | Promise<unknown>;
};

type PendingRequest = {
	resolve(value: Needle2WorkerResult): void;
	reject(reason: unknown): void;
};

export class Needle2WorkerBackend implements Needle2Backend {
	readonly provider = "worker" as const;

	private readonly pending = new Map<number, PendingRequest>();
	private nextRequestID = 0;
	private disposed = false;
	private disposePromise: Promise<void> | undefined;

	static async create(
		wasm: Needle2BinarySource,
		weights: Needle2BinarySource,
		options?: WorkerOptions,
		engine: "wasm" | "native" = "wasm",
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
		connection.onError((error) => this.failPending(error));
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

	dispose(): Promise<void> {
		this.disposePromise ??= this.disposeInternal();
		return this.disposePromise;
	}

	private async disposeInternal(): Promise<void> {
		if (this.disposed) return;
		try {
			await this.request({ operation: "dispose" });
		} finally {
			this.disposed = true;
			await this.connection.terminate();
			this.failPending(
				new Needle2Error("disposed", "This Needle 2 runtime has been disposed."),
			);
		}
	}

	private request(
		request: Needle2WorkerRequestInput,
	): Promise<Needle2WorkerResult> {
		if (this.disposed) {
			return Promise.reject(
				new Needle2Error("disposed", "This Needle 2 runtime has been disposed."),
			);
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
}

async function workerConnection(
	url: URL,
	options?: WorkerOptions,
): Promise<WorkerConnection> {
	if (isBrowserEnvironment() && typeof Worker !== "undefined") {
		return browserWorkerConnection(url, options);
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
		onError: (handler) => worker.on("error", handler),
		terminate: () => worker.terminate(),
	};
}

function browserWorkerConnection(
	url: URL,
	options?: WorkerOptions,
): WorkerConnection {
	const worker = new Worker(url, { ...options, type: "module" });
	return {
		postMessage: (message) => worker.postMessage(message),
		onMessage: (handler) =>
			worker.addEventListener("message", (event) =>
				handler(event.data as Needle2WorkerResponse),
			),
		onError: (handler) =>
			worker.addEventListener("error", (event) =>
				handler(new Error(event.message)),
			),
		terminate: () => worker.terminate(),
	};
}

function deserializeError(error: Needle2SerializedError): Error {
	const result = new Needle2Error(error.code ?? "worker-error", error.message);
	result.name = error.name;
	if (error.stack) result.stack = error.stack;
	return result;
}
