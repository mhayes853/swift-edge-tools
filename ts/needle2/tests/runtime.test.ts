import { afterEach, describe, expect, test } from "vitest";
import {
  needle2Runtime,
  type Needle2Initialization,
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
      expect(result).toMatchSnapshot({
        metrics: {
          prefillTokensPerSecond: expect.any(Number),
          decodeTokensPerSecond: expect.any(Number),
          peakRAMMegabytes: expect.any(Number)
        }
      });
    });
  }
);
