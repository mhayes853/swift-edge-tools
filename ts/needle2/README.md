# Needle 2 JavaScript Runtime

Needle 2 can run in a dedicated Web Worker or directly in the current JavaScript realm. The
runtime automatically loads the bundled engine and default weights.

```ts
import { needle2 } from "@edge-tools/needle2";

const runtime = await needle2({ provider: "worker" });
const result = await runtime.generate({
  prompt: "Dim the living room lights to 30 percent.",
  initialization: {
    systemValues: {
      assistant: "Needle"
    },
    tools: [
      {
        name: "set_lights",
        description: "Controls the lights.",
        parameters: {
          type: "object",
          properties: {
            room: { type: "string" },
            brightness: { type: "integer" }
          },
          required: ["room", "brightness"]
        }
      }
    ]
  }
});
```

Use `{ provider: "direct" }` when the calling JavaScript realm may block during generation. Both
providers expose the same asynchronous API. If `systemValues` is omitted, no system facts are
included. Use `defaultSystemValues` to collect environment values explicitly, then pass the result
as `systemValues`. `defaultSystemPrompt` also formats values directly when needed; arbitrary fact
keys are supported.

The standalone bundle installs `needle2` directly:

```html
<script src="https://example.com/needle2.min.js"></script>
<script>
  const runtime = await needle2({ provider: "worker" });
</script>
```

`wasm`, `weights`, and the direct provider's `factory` option allow applications to override the
bundled resources. The worker provider always uses the bundled worker implementation. Pass a
`.cact` file through `weights`, or use `runtime.load(...)` to replace the weights after
initialization.

Give a tool a `call` handler and `generate` invokes it for you, attaching the result to that tool
call as `output`. Handlers for a single generation all run in parallel, and the return types flow
through, so `output` is typed per tool:

```ts
const result = await runtime.generate({
  prompt: "What's the weather in Paris?",
  initialization: {
    tools: [
      {
        name: "get_weather",
        description: "Get the weather for a city.",
        parameters: {
          type: "object",
          properties: { city: { type: "string" } },
          required: ["city"]
        },
        call: async (args: { city: string }) => ({ celsius: 21 })
      }
    ]
  }
});

if (result.success) {
  for (const call of result.functionCalls) {
    if (call.name === "get_weather") {
      console.log(call.output.celsius); // number
    }
  }
}
```

Only the `name`, `description`, and `parameters` of each tool reach the engine, so handlers are free
to close over anything and still work under the worker provider. Tools declared without a `call`
behave exactly as before and produce no `output`. Handlers never run for a failed generation, which
is why `output` is reachable only after narrowing on `success`. If a handler throws, `generate`
rejects with a `Needle2ToolCallError` carrying every failure and the generation that produced them.

`runLoop` drives the whole loop for you, feeding each turn's tool responses back as the next
turn's prompt until the model answers:

```ts
const response = await runtime.runLoop({
  prompt: "What's the weather in Paris, and email blob@gmail.com about it?",
  initialization: { tools },
  maximumTurns: 8
});

response.terminationCause; // "responded"
response.steps[0].toolResponses; // [{ celsius: 21 }, { status: "sent" }]
```

Each step holds that turn's `generation` and the `toolResponses` fed back after it. Outputs live in
`toolResponses` rather than on the calls, because a tool that fails contributes an error response
instead of an output. `runLoop` does not throw when a handler fails: it sends `{ "error": "..." }`
back as that call's response so the model can recover on the next turn, which is also what a tool
declared without a `call` produces. This matches the official Python `run`. `generate` still throws
in that situation, because it has no next turn to put the error in.

The loop ends with a `terminationCause` of `"responded"`, `"refused"`, `"no-tool-calls"`,
`"maximum-turns-reached"`, or `"failed"`. Note that a `"responded"` turn carries no answer text —
Needle 2 signals that it is done rather than writing prose, so `reasoning` is the only content.
`maximumTurns` defaults to 8 and must be between 1 and 8.

Each runtime keeps the underlying Needle 2 conversation alive after `generate`, so the loop can
also be driven manually. `generateRaw` produces one turn without invoking any handlers, leaving the
calls for you to execute and feed back as the next prompt:

```ts
const generation = await runtime.generateRaw({ prompt, initialization: { tools } });
const outputs = generation.functionCalls.map((call) => run(call));
const next = await runtime.generateRaw({ prompt: JSON.stringify(outputs), initialization: { tools } });
```

Call
`runtime.reset()` when the conversation is complete or before changing the initialization. Direct
runtimes share one process-wide native model, so another direct runtime must wait until the active
runtime is reset or disposed.

Run the full package test suite with `npm test`. It builds the package, runs the Node and browser
Vitest suites, and then runs the direct and worker providers under both Deno and Bun. The Deno
runtime test needs read permission for the bundled WASM and model assets; its command is
`npm run test:deno` (equivalent to `deno test --allow-read ...`). Use `npm run test:bun` to run only
the Bun tests.

Native bindings can be built from the distributed static library with
`npm run native:build`. The installed package includes the build script and native wrapper source;
the script downloads only the current host's pinned, checksum-verified static library and writes the
result to `dist/native`. When `engine: "native"` is selected, the runtime locates the resulting
native addon or shared library automatically. The build currently supports Apple silicon macOS and
x86-64 or ARM64 Linux, and requires `curl`, Clang/C++, and Node development headers:

```ts
import { needle2 } from "@edge-tools/needle2";

const runtime = await needle2({
  provider: "direct",
  engine: "native"
});
```

Node uses the N-API addon; Deno and Bun use the C ABI shared library. `npm run test:native` tests
the Node addon, while `npm run test:native-library` tests the shared library through Deno and Bun.
Native generation uses the same process-global serialization as the WASM direct backend because the
underlying C engine is global-state based. Use `engine: "auto"` to prefer an available native build
and fall back to WASM when native loading is unavailable.

The wrapper is MIT licensed. The redistributed Needle 2 engine, WASM, and default weights are
provided by Cactus Compute under Apache-2.0; its license is included as `LICENSE-Needle2`.
