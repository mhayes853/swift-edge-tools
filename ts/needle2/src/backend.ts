// @ts-expect-error The pinned upstream Emscripten binding does not publish declarations.
import bundledFactory from "../vendor/needle.mjs";
import {
  binarySourceBytes,
  isBrowserEnvironment,
  Needle2Error,
  PromiseQueue,
  serializeBinarySource,
  type Needle2SerializedBinarySource
} from "./internal";
import type {
  Needle2Backend,
  Needle2BinarySource,
  Needle2Factory,
  Needle2GenerateOptions,
  Needle2NativeGeneration
} from "./types";

type Needle2EmscriptenModule = {
  readonly HEAPU8: Uint8Array;
  readonly _needle_load: (pointer: number, length: bigint) => number;
  readonly _needle_reset: () => void;
  readonly _malloc: (capacity: number) => number;
  readonly _free: (pointer: number) => void;
  readonly UTF8ToString: (pointer: number) => string;
  readonly ccall: (
    name: string,
    returnType: "number",
    argumentTypes: readonly string[],
    argumentValues: readonly unknown[]
  ) => number;
};

export class Needle2DirectBackend implements Needle2Backend {
  readonly provider = "direct" as const;

  private module: Needle2EmscriptenModule | undefined;
  private readonly operations = new PromiseQueue();
  private initializationFingerprint: string | undefined;
  private weightsPointer: number | undefined;
  private disposed = false;

  static async create(
    wasm: Needle2BinarySource,
    weights: Needle2BinarySource,
    factory: Needle2Factory = bundledFactory as Needle2Factory
  ): Promise<Needle2DirectBackend> {
    const [wasmBytes, weightBytes] = await Promise.all([
      binarySourceBytes(wasm),
      binarySourceBytes(weights)
    ]);
    const module = emscriptenModule(await factory({ wasmBinary: wasmBytes }));
    const backend = new Needle2DirectBackend(module);
    backend.loadImmediately(weightBytes);
    return backend;
  }

  private constructor(module: Needle2EmscriptenModule) {
    this.module = module;
  }

  generate(options: Needle2GenerateOptions): Promise<Needle2NativeGeneration> {
    return this.operations.enqueue(() => {
      this.requireActive();
      const module = this.requireModule();
      const maxTokens = positiveInteger(options.maxTokens ?? 256, "maxTokens");
      const outputCapacity = positiveInteger(
        options.outputCapacity ?? 65_536,
        "outputCapacity"
      );
      const initializationFingerprint = JSON.stringify(options.initialization);
      if (initializationFingerprint !== this.initializationFingerprint) {
        this.initialize(options);
        this.initializationFingerprint = initializationFingerprint;
      }

      const output = module._malloc(outputCapacity);
      if (output === 0) {
        throw new Needle2Error("allocation-failed", "Needle 2 could not allocate its output buffer.");
      }
      try {
        const tokenCount = module.ccall(
          "needle_complete",
          "number",
          ["string", "number", "number", "number"],
          [options.prompt, maxTokens, output, outputCapacity]
        );
        if (tokenCount < 0) {
          throw new Needle2Error(
            "generation-failed",
            `needle_complete failed with status ${tokenCount}.`
          );
        }
        return {
          json: module.UTF8ToString(output),
          tokenCount
        };
      } finally {
        module._free(output);
        module._needle_reset();
      }
    });
  }

  load(weights: Needle2BinarySource): Promise<void> {
    return this.operations.enqueue(async () => {
      this.requireActive();
      this.loadImmediately(await binarySourceBytes(weights));
    });
  }

  dispose(): Promise<void> {
    return this.operations.enqueue(() => {
      if (this.weightsPointer) {
        this.module?._free(this.weightsPointer);
      }
      this.disposed = true;
      this.initializationFingerprint = undefined;
      this.weightsPointer = undefined;
      this.module = undefined;
    });
  }

  private initialize(options: Needle2GenerateOptions): void {
    const module = this.requireModule();
    const initialization = options.initialization;
    const result = module.ccall(
      "needle_init",
      "number",
      ["string", "string", "string"],
      [
        initialization.systemPrompt,
        JSON.stringify(initialization.tools),
        initialization.toolIndexPath ?? null
      ]
    );
    if (result < 0) {
      throw new Needle2Error(
        "initialization-failed",
        `needle_init failed with status ${result}.`
      );
    }
  }

  private loadImmediately(weights: Uint8Array): void {
    const module = this.requireModule();
    const pointer = module._malloc(weights.byteLength);
    if (pointer === 0) {
      throw new Needle2Error("allocation-failed", "Needle 2 could not allocate its weights buffer.");
    }
    module.HEAPU8.set(weights, pointer);
    const result = module._needle_load(pointer, BigInt(weights.byteLength));
    if (result < 0) {
      module._free(pointer);
      throw new Needle2Error("loading-failed", `needle_load failed with status ${result}.`);
    }
    if (this.weightsPointer) {
      module._free(this.weightsPointer);
    }
    this.weightsPointer = pointer;
    this.initializationFingerprint = undefined;
  }

  private requireActive(): void {
    if (this.disposed) {
      throw new Needle2Error("disposed", "This Needle 2 runtime has been disposed.");
    }
  }

  private requireModule(): Needle2EmscriptenModule {
    this.requireActive();
    return this.module!;
  }
}

export type Needle2WorkerRequest =
  | {
      id: number;
      operation: "initialize";
      wasm: Needle2SerializedBinarySource;
      weights: Needle2SerializedBinarySource;
    }
  | { id: number; operation: "generate"; options: Needle2GenerateOptions }
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
  revoke(): void;
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

  static async create(
    workerURL: URL,
    wasm: Needle2BinarySource,
    weights: Needle2BinarySource,
    options?: WorkerOptions
  ): Promise<Needle2WorkerBackend> {
    const backend = new Needle2WorkerBackend(await workerConnection(workerURL, options));
    await backend.request({
      operation: "initialize",
      wasm: serializeBinarySource(wasm),
      weights: serializeBinarySource(weights)
    });
    return backend;
  }

  private constructor(private readonly connection: WorkerConnection) {
    connection.onMessage(message => this.receive(message));
    connection.onError(error => this.failPending(error));
  }

  async generate(options: Needle2GenerateOptions): Promise<Needle2NativeGeneration> {
    const result = await this.request({ operation: "generate", options });
    if (!result) {
      throw new Needle2Error("worker-protocol", "The Needle 2 worker returned no generation.");
    }
    return result;
  }

  async load(weights: Needle2BinarySource): Promise<void> {
    await this.request({ operation: "load", weights: serializeBinarySource(weights) });
  }

  async dispose(): Promise<void> {
    if (this.disposed) {
      return;
    }
    try {
      await this.request({ operation: "dispose" });
    } finally {
      this.disposed = true;
      await this.connection.terminate();
      this.connection.revoke();
      this.failPending(new Needle2Error("disposed", "This Needle 2 runtime has been disposed."));
    }
  }

  private request(request: Needle2WorkerRequestInput): Promise<Needle2WorkerResult> {
    if (this.disposed) {
      return Promise.reject(
        new Needle2Error("disposed", "This Needle 2 runtime has been disposed.")
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
    if (!pending) {
      return;
    }
    this.pending.delete(response.id);
    if (response.success) {
      pending.resolve(response.result);
    } else {
      pending.reject(deserializeError(response.error));
    }
  }

  private failPending(error: Error): void {
    for (const pending of this.pending.values()) {
      pending.reject(error);
    }
    this.pending.clear();
  }
}

function emscriptenModule(value: unknown): Needle2EmscriptenModule {
  if (typeof value !== "object" || value === null) {
    throw invalidModule();
  }
  const module = value as Partial<Needle2EmscriptenModule>;
  if (
    !(module.HEAPU8 instanceof Uint8Array) ||
    typeof module._needle_load !== "function" ||
    typeof module._needle_reset !== "function" ||
    typeof module._malloc !== "function" ||
    typeof module._free !== "function" ||
    typeof module.UTF8ToString !== "function" ||
    typeof module.ccall !== "function"
  ) {
    throw invalidModule();
  }
  return module as Needle2EmscriptenModule;
}

function invalidModule(): Needle2Error {
  return new Needle2Error(
    "invalid-module",
    "The Needle 2 factory returned an object without the expected Emscripten exports."
  );
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0 || value > 2_147_483_647) {
    throw new Needle2Error(
      "invalid-generate-options",
      `Needle 2 ${name} must be an integer between 1 and 2147483647.`
    );
  }
  return value;
}

async function workerConnection(url: URL, options?: WorkerOptions): Promise<WorkerConnection> {
  if (isBrowserEnvironment() && typeof Worker !== "undefined") {
    return browserWorkerConnection(url, options);
  }

  const workerThreadsSpecifier = "node:worker_threads";
  const { Worker: NodeWorker } = await import(/* @vite-ignore */ workerThreadsSpecifier);
  const worker = new NodeWorker(url, options?.name ? { name: options.name } : undefined);
  return {
    postMessage: message => worker.postMessage(message),
    onMessage: handler => worker.on("message", handler),
    onError: handler => worker.on("error", handler),
    terminate: () => worker.terminate(),
    revoke() {}
  };
}

function browserWorkerConnection(url: URL, options?: WorkerOptions): WorkerConnection {
  let workerURL = url;
  let objectURL: string | undefined;
  if (url.origin !== location.origin) {
    objectURL = URL.createObjectURL(
      new Blob([`import ${JSON.stringify(url.href)};`], { type: "text/javascript" })
    );
    workerURL = new URL(objectURL);
  }
  const worker = new Worker(workerURL, { ...options, type: "module" });
  return {
    postMessage: message => worker.postMessage(message),
    onMessage: handler =>
      worker.addEventListener("message", event => handler(event.data as Needle2WorkerResponse)),
    onError: handler =>
      worker.addEventListener("error", event => handler(new Error(event.message))),
    terminate: () => worker.terminate(),
    revoke() {
      if (objectURL) {
        URL.revokeObjectURL(objectURL);
      }
    }
  };
}

function deserializeError(error: Needle2SerializedError): Error {
  const result = new Needle2Error(error.code ?? "worker-error", error.message);
  result.name = error.name;
  if (error.stack) {
    result.stack = error.stack;
  }
  return result;
}
