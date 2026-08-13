# Needle 2 JavaScript Runtime

Needle 2 can run in a dedicated Web Worker or directly in the current JavaScript realm. The
runtime automatically loads the bundled engine and default weights.

```ts
import { needle2Runtime } from "@edge-tools/needle2";

const runtime = await needle2Runtime({ provider: "worker" });
const result = await runtime.generate({
  prompt: "Dim the living room lights to 30 percent.",
  initialization: {
    systemFactsOptions: {
      overrides: { assistant: "Needle" }
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
providers expose the same asynchronous API. If `systemPrompt` is omitted, the runtime formats
`systemFacts` (or facts collected by `defaultSystemFacts`) automatically. Use
`defaultSystemPrompt` to format facts yourself; arbitrary fact keys are supported. Location is
omitted by default and can be enabled with `includeLocation: true` or supplied through a provider.

The standalone bundle installs `needle2Runtime` directly:

```html
<script src="https://example.com/needle2.min.js"></script>
<script>
  const runtime = await needle2Runtime({ provider: "worker" });
</script>
```

`wasm`, `weights`, `workerURL`, and `factory` options allow applications to override the bundled
resources. Pass a `.cact` file through `weights`, or use `runtime.load(...)` to replace the weights
after initialization.

`isBrowserEnvironment()` and `isNodeLikeEnvironment()` are available when applications need to
choose behavior outside the runtime's automatic environment detection.

The wrapper is MIT licensed. The redistributed Needle 2 engine, WASM, and default weights are
provided by Cactus Compute under Apache-2.0; its license is included as `LICENSE-Needle2`.
