import {
  Needle2DirectBackend,
  type Needle2SerializedError,
  type Needle2WorkerRequest,
  type Needle2WorkerResponse,
  type Needle2WorkerResult
} from "./backend";
import { PromiseQueue } from "./internal";
import type { Needle2Backend } from "./types";

type WorkerPort = {
  postMessage(message: Needle2WorkerResponse): void;
  onMessage(handler: (message: Needle2WorkerRequest) => void): void;
};

const port = await workerPort();
let backend: Needle2Backend | undefined;
const operations = new PromiseQueue();

port.onMessage(request => {
  void operations.enqueue(() => processRequest(request));
});

async function processRequest(request: Needle2WorkerRequest): Promise<void> {
  try {
    const result = await handle(request);
    port.postMessage({
      id: request.id,
      success: true,
      ...(result === undefined ? {} : { result })
    });
  } catch (error) {
    port.postMessage({
      id: request.id,
      success: false,
      error: serializeError(error)
    });
  }
}

async function handle(request: Needle2WorkerRequest): Promise<Needle2WorkerResult> {
  switch (request.operation) {
    case "initialize":
      if (backend) {
        throw new Error("The Needle 2 worker is already initialized.");
      }
      backend = await Needle2DirectBackend.create(request.wasm, request.weights);
      return undefined;

    case "generate":
      return requireBackend().generate(request.options);

    case "load":
      await requireBackend().load(request.weights);
      return undefined;

    case "dispose":
      await requireBackend().dispose();
      backend = undefined;
      return undefined;
  }
}

function requireBackend(): Needle2Backend {
  if (!backend) {
    throw new Error("The Needle 2 worker is not initialized.");
  }
  return backend;
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
        globalThis.addEventListener("message", event =>
          handler((event as MessageEvent<Needle2WorkerRequest>).data)
        );
      }
    };
  }

  const workerThreadsSpecifier = "node:worker_threads";
  const { parentPort } = await import(/* @vite-ignore */ workerThreadsSpecifier);
  if (!parentPort) {
    throw new Error("Needle 2 could not access its worker message port.");
  }
  return {
    postMessage(message) {
      parentPort.postMessage(message);
    },
    onMessage(handler) {
      parentPort.on("message", handler);
    }
  };
}

function serializeError(value: unknown): Needle2SerializedError {
  if (value instanceof Error) {
    const error: Needle2SerializedError = {
      name: value.name,
      message: value.message
    };
    if (value.stack) {
      error.stack = value.stack;
    }
    if ("code" in value && typeof value.code === "string") {
      error.code = value.code;
    }
    return error;
  }
  return {
    name: "Error",
    message: String(value)
  };
}
