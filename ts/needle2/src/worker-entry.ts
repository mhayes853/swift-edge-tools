import type {
	Needle2SerializedError,
	Needle2WorkerRequest,
	Needle2WorkerResponse,
	Needle2WorkerResult,
} from "./backend.js";
import { nativeBinding, wasmBinding } from "./bindings.js";
import { binarySourceBytes, Needle2Error } from "./internal.js";
import type { Needle2Binding } from "./bindings.js";
import type { Needle2Engine, Needle2ResolvedInitialization } from "./runtime.js";
import type { Needle2SerializedBinarySource } from "./internal.js";

type WorkerPort = {
	postMessage(message: Needle2WorkerResponse): void;
	onMessage(handler: (message: Needle2WorkerRequest) => void): void;
};

const port = await workerPort();
let binding: Needle2Binding | undefined;
let initialization: Needle2ResolvedInitialization | undefined;
let initialized = false;

port.onMessage((request) => {
	void processRequest(request);
});

async function processRequest(request: Needle2WorkerRequest): Promise<void> {
	try {
		const result = await handle(request);
		port.postMessage({
			success: true,
			...(result === undefined ? {} : { result }),
		});
	} catch (error) {
		port.postMessage({
			success: false,
			error: serializeError(error),
		});
	}
}

async function handle(
	request: Needle2WorkerRequest,
): Promise<Needle2WorkerResult> {
	switch (request.operation) {
		case "initialize":
			binding = await loadedBinding(
				request.engine,
				request.wasm,
				request.weights,
			);
			initialization = request.initialization;
			return undefined;

		case "generate":
			if (!initialized) {
				binding!.initialize(initialization!);
				initialized = true;
			}
			return binding!.complete(request.options);

		case "load":
			if (initialized) {
				throw new Needle2Error(
					"active-session",
					"Reset the active Needle 2 runtime before loading new weights.",
				);
			}
			binding!.load(await binarySourceBytes(request.weights));
			return undefined;

		case "reset":
			if (initialized) {
				binding!.reset();
			}
			initialized = false;
			return undefined;

		case "dispose":
			if (initialized) {
				binding!.reset();
			}
			binding = undefined;
			initialization = undefined;
			initialized = false;
			return undefined;
	}
}

async function loadedBinding(
	engine: Needle2Engine,
	wasm: Needle2SerializedBinarySource,
	weights: Needle2SerializedBinarySource,
): Promise<Needle2Binding> {
	const weightBytes = await binarySourceBytes(weights);
	if (engine === "native") {
		return loadedNativeBinding(weightBytes);
	}
	if (engine === "auto") {
		const binding = await loadedNativeBinding(weightBytes).catch(
			() => undefined,
		);
		if (binding) {
			return binding;
		}
	}
	const binding = await wasmBinding(await binarySourceBytes(wasm));
	binding.load(weightBytes);
	return binding;
}

async function loadedNativeBinding(
	weights: Uint8Array,
): Promise<Needle2Binding> {
	const binding = await nativeBinding();
	binding.load(weights);
	return binding;
}

async function workerPort(): Promise<WorkerPort> {
	if (
		typeof globalThis.addEventListener === "function" &&
		typeof globalThis.postMessage === "function"
	) {
		return {
			postMessage(message) {
				globalThis.postMessage(message);
			},
			onMessage(handler) {
				globalThis.addEventListener("message", (event) =>
					handler((event as MessageEvent<Needle2WorkerRequest>).data),
				);
			},
		};
	}

	const workerThreadsSpecifier = "node:worker_threads";
	const { parentPort } = await import(/* @vite-ignore */ workerThreadsSpecifier);
	if (!parentPort)
		throw new Error("Needle 2 could not access its worker message port.");

	return {
		postMessage(message) {
			parentPort.postMessage(message);
		},
		onMessage(handler) {
			parentPort.on("message", handler);
		},
	};
}

function serializeError(value: unknown): Needle2SerializedError {
	if (value instanceof Error) {
		const error: Needle2SerializedError = {
			name: value.name,
			message: value.message,
		};
		if (value.stack) {
			error.stack = value.stack;
		}
		if ("code" in value && typeof value.code === "string") {
			error.code = value.code;
		}
		return error;
	}
	return { name: "Error", message: String(value) };
}
