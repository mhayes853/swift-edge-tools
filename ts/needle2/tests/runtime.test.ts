import { afterEach, describe, expect, test } from "vitest";
import {
  needle2Runtime,
  type Needle2Initialization,
  type Needle2Factory,
  type Needle2Provider,
  type Needle2Runtime
} from "../dist/index.js";

const runtimes: Needle2Runtime[] = [];
const emailInitialization: Needle2Initialization = {
  systemPrompt: "",
  tools: [
    {
      name: "send_email",
      description: "Sends an email to a recipient with an email address.",
      parameters: {
        type: "object",
        properties: {
          address: {
            type: "string",
            pattern: "[a-z][a-z0-9]{1,10}@gmail\\.com",
            description: "The recipient's email address.",
            examples: ["blob@gmail.com"]
          },
          subject: {
            type: "string",
            description: "The subject of an email."
          },
          body: {
            type: "string",
            description: "The content of an email."
          }
        },
        required: ["address", "subject", "body"]
      }
    }
  ]
};

afterEach(async () => {
  await Promise.all(runtimes.splice(0).map(runtime => runtime.dispose()));
});

test("returns the native error response for truncated generation", async () => {
  const runtime = await needle2Runtime({ provider: "direct" });
  runtimes.push(runtime);

  const result = await runtime.generate({
    prompt: "Send an email to blob@gmail.com asking them to go hiking.",
    initialization: emailInitialization,
    maxTokens: 4
  });

  expect(result.success).toBe(false);
  expect(result.type).toBe("call");
  expect(result.tokenCount).toBe(4);
  if (result.success) {
    throw new Error("Expected Needle 2 generation to be truncated.");
  }
  expect(result.error).toBe("tool call truncated: token budget exhausted");
  expect(result.errorCode).toBe("truncated");
});

test("isolates multiple direct runtimes sharing one native module", async () => {
  const fake = fakeNeedleFactory();
  const weights = new Uint8Array([1]);
  const thermostat = await needle2Runtime({
    provider: "direct",
    factory: fake.factory,
    wasm: new Uint8Array([1]),
    weights
  });
  const weather = await needle2Runtime({
    provider: "direct",
    factory: fake.factory,
    wasm: new Uint8Array([1]),
    weights
  });
  runtimes.push(thermostat, weather);

  const thermostatInitialization = initializationFor("set_thermostat");
  const weatherInitialization = initializationFor("get_weather");
  const thermostatPrompt = { prompt: "set it to 21 degrees", initialization: thermostatInitialization };
  const weatherPrompt = { prompt: "what's the weather in Paris?", initialization: weatherInitialization };

  expect((await thermostat.generate(thermostatPrompt)).functionCalls[0]?.name).toBe("set_thermostat");
  expect((await weather.generate(weatherPrompt)).functionCalls[0]?.name).toBe("get_weather");
  expect((await thermostat.generate(thermostatPrompt)).functionCalls[0]?.name).toBe("set_thermostat");
  expect(fake.factoryCalls).toBe(1);
});

describe.each(["direct", "worker"] satisfies Needle2Provider[])(
  "Needle2Runtime with the %s provider",
  provider => {
    test("generates a parsed response", async () => {
      const runtime = await needle2Runtime({ provider });
      runtimes.push(runtime);

      const result = await runtime.generate({
        prompt: "Send an email to blob@gmail.com asking them to go hiking.",
        initialization: emailInitialization
      });

      expect(result.success).toBe(true);
      expect(result.type).toBe("call");
      expect(result.tokenCount).toBeGreaterThan(0);
      expect(result.functionCalls).toHaveLength(1);
      expect(result.functionCalls[0]).toMatchObject({
        name: "send_email",
        arguments: { address: "blob@gmail.com" }
      });
      expect(result.metrics.peakRAMMegabytes).toBeUndefined();
      expect(result).toMatchSnapshot({
        metrics: {
          prefillTokensPerSecond: expect.any(Number),
          decodeTokensPerSecond: expect.any(Number)
        }
      });
    });
  }
);

function initializationFor(toolName: string): Needle2Initialization {
  return {
    systemPrompt: "",
    tools: [{ name: toolName, parameters: { type: "object" } }]
  };
}

function fakeNeedleFactory(): {
  factory: Needle2Factory;
  factoryCalls: number;
} {
  const module = {
    HEAPU8: new Uint8Array(1024),
    activeTool: "",
    factoryCalls: 0,
    loadCalls: 0,
    nextPointer: 1,
    _needle_load(pointer: number, _length: bigint) {
      void pointer;
      module.loadCalls += 1;
      return 0;
    },
    _needle_reset() {},
    _malloc(capacity: number) {
      const pointer = module.nextPointer;
      module.nextPointer += capacity;
      return pointer;
    },
    _free(_pointer: number) {},
    UTF8ToString(_pointer: number) {
      return JSON.stringify({
        type: "call",
        success: true,
        function_calls: [{ name: module.activeTool, arguments: {} }]
      });
    },
    ccall(
      name: string,
      _returnType: "number",
      _argumentTypes: readonly string[],
      argumentValues: readonly unknown[]
    ) {
      if (name === "needle_init") {
        const tools = JSON.parse(String(argumentValues[1])) as Array<{ name: string }>;
        module.activeTool = tools[0]?.name ?? "";
      }
      return 1;
    }
  };
  const factory: Needle2Factory = async () => {
    module.factoryCalls += 1;
    return module;
  };
  return {
    factory,
    get factoryCalls() {
      return module.factoryCalls;
    }
  };
}
