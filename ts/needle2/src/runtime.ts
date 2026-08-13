import { Needle2DirectBackend, Needle2WorkerBackend } from "./backend";
import { defaultAssetURL, Needle2ProtocolError, setAssetBaseURL } from "./internal";
import type {
  Needle2Backend,
  Needle2BinarySource,
  Needle2GenerateOptions,
  Needle2GenerationResult,
  Needle2JSONObject,
  Needle2Provider,
  Needle2ResponseType,
  Needle2RuntimeOptions
} from "./types";

setAssetBaseURL(new URL(import.meta.url));

export class Needle2Runtime {
  readonly provider: Needle2Provider;

  private constructor(private readonly backend: Needle2Backend) {
    this.provider = backend.provider;
  }

  /** @internal */
  static async create(options: Needle2RuntimeOptions): Promise<Needle2Runtime> {
    const wasm = options.wasm ?? defaultAssetURL("needle.wasm");
    const weights = options.weights ?? defaultAssetURL("needle2.cact");
    if (options.provider === "direct") {
      return new Needle2Runtime(
        await Needle2DirectBackend.create(wasm, weights, options.factory)
      );
    }

    const workerURL = options.workerURL
      ? new URL(options.workerURL, defaultAssetURL("./"))
      : defaultAssetURL("needle2.worker.mjs");
    return new Needle2Runtime(
      await Needle2WorkerBackend.create(workerURL, wasm, weights, options.workerOptions)
    );
  }

  async generate(options: Needle2GenerateOptions): Promise<Needle2GenerationResult> {
    const generation = await this.backend.generate(options);
    return parseGenerationResult(generation.json, generation.tokenCount);
  }

  load(weights: Needle2BinarySource): Promise<void> {
    return this.backend.load(weights);
  }

  dispose(): Promise<void> {
    return this.backend.dispose();
  }
}

export function needle2Runtime(options: Needle2RuntimeOptions): Promise<Needle2Runtime> {
  return Needle2Runtime.create(options);
}

function parseGenerationResult(json: string, tokenCount: number): Needle2GenerationResult {
  let response: Record<string, unknown>;
  try {
    response = JSON.parse(json) as Record<string, unknown>;
  } catch (cause) {
    throw new Needle2ProtocolError("Needle 2 returned malformed JSON.", { cause });
  }
  if (typeof response.type !== "string" || typeof response.success !== "boolean") {
    throw new Needle2ProtocolError("Needle 2 returned an invalid response.");
  }

  const functionCalls = Array.isArray(response.function_calls)
    ? response.function_calls.map(value => {
        const call = value as { name?: unknown; arguments?: unknown };
        return {
          name: String(call.name ?? ""),
          arguments: (call.arguments ?? {}) as Needle2JSONObject
        };
      })
    : [];
  const common = {
    type: response.type as Needle2ResponseType,
    functionCalls,
    tokenCount,
    metrics: {
      ...numberProperty("prefillTokensPerSecond", response.prefill_tps),
      ...numberProperty("decodeTokensPerSecond", response.decode_tps),
      ...positiveNumberProperty("peakRAMMegabytes", response.peak_ram_mb)
    },
    ...stringProperty("reasoning", response.reasoning),
    ...numberProperty("confidence", response.confidence)
  };
  if (response.success) {
    return { success: true, ...common };
  }
  return {
    success: false,
    ...common,
    error: typeof response.error === "string" ? response.error : "Needle 2 generation failed.",
    ...stringProperty("errorCode", response.error_code)
  };
}

function stringProperty<Key extends string>(
  key: Key,
  value: unknown
): { [Property in Key]?: string } {
  return typeof value === "string" ? ({ [key]: value } as never) : {};
}

function numberProperty<Key extends string>(
  key: Key,
  value: unknown
): { [Property in Key]?: number } {
  return typeof value === "number" ? ({ [key]: value } as never) : {};
}

function positiveNumberProperty<Key extends string>(
  key: Key,
  value: unknown
): { [Property in Key]?: number } {
  return typeof value === "number" && value > 0 ? ({ [key]: value } as never) : {};
}
