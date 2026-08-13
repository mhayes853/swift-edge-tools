import type { Needle2BinarySource } from "./types";

export type Needle2SerializedBinarySource = string | Uint8Array;

let assetBaseURL: URL | undefined;

export class Needle2Error extends Error {
  readonly code: string;

  constructor(code: string, message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "Needle2Error";
    this.code = code;
  }
}

export class Needle2ProtocolError extends Needle2Error {
  constructor(message: string, options?: ErrorOptions) {
    super("invalid-response", message, options);
    this.name = "Needle2ProtocolError";
  }
}

export class PromiseQueue {
  private tail: Promise<void> = Promise.resolve();

  enqueue<Result>(operation: () => Result | PromiseLike<Result>): Promise<Result> {
    const result = this.tail.then(operation, operation);
    this.tail = result.then(
      () => undefined,
      () => undefined
    );
    return result;
  }
}

export function isBrowserEnvironment(): boolean {
  return (
    typeof globalThis.document !== "undefined" ||
    typeof globalThis.WorkerGlobalScope !== "undefined"
  );
}

export function isNodeLikeEnvironment(): boolean {
  return !isBrowserEnvironment();
}

export function setAssetBaseURL(url: URL): void {
  assetBaseURL = url;
}

export function defaultAssetURL(name: string): URL {
  if (!assetBaseURL) {
    throw new Error("Needle 2 could not determine its default asset URL.");
  }
  return new URL(name, assetBaseURL);
}

export function serializeBinarySource(
  source: Needle2BinarySource
): Needle2SerializedBinarySource {
  if (typeof source === "string") {
    return source;
  }
  if (source instanceof URL) {
    return source.href;
  }
  if (source instanceof Uint8Array) {
    return source.slice();
  }
  return new Uint8Array(source.slice(0));
}

export async function binarySourceBytes(
  source: Needle2BinarySource | Needle2SerializedBinarySource
): Promise<Uint8Array> {
  if (source instanceof Uint8Array) {
    return source;
  }
  if (source instanceof ArrayBuffer) {
    return new Uint8Array(source);
  }

  const url = source instanceof URL ? source : new URL(source, fallbackBaseURL());
  if (url.protocol === "file:") {
    if (!isNodeLikeEnvironment()) {
      throw new Error(`The ${url.href} file URL is unavailable outside a Node-like environment.`);
    }
    const fileSystemSpecifier = "node:fs/promises";
    const { readFile } = await import(/* @vite-ignore */ fileSystemSpecifier);
    return new Uint8Array(await readFile(url));
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url.href}: HTTP ${response.status}.`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

function fallbackBaseURL(): URL {
  return typeof location === "undefined" ? new URL("file:///") : new URL(location.href);
}
