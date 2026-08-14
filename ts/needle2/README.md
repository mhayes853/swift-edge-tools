# Needle 2 JavaScript Runtime

Needle 2 can run in a dedicated Web Worker or directly in the current JavaScript realm. The
runtime automatically loads the bundled engine and default weights.

```ts
import { needle2Runtime } from "@edge-tools/needle2";

const runtime = await needle2Runtime({ provider: "worker" });
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

The standalone bundle installs `needle2Runtime` directly:

```html
<script src="https://example.com/needle2.min.js"></script>
<script>
  const runtime = await needle2Runtime({ provider: "worker" });
</script>
```

`wasm`, `weights`, and the direct provider's `factory` option allow applications to override the
bundled resources. The worker provider always uses the bundled worker implementation. Pass a
`.cact` file through `weights`, or use `runtime.load(...)` to replace the weights after
initialization.

`isBrowserEnvironment()` and `isNodeLikeEnvironment()` are available when applications need to
choose behavior outside the runtime's automatic environment detection.

The wrapper is MIT licensed. The redistributed Needle 2 engine, WASM, and default weights are
provided by Cactus Compute under Apache-2.0; its license is included as `LICENSE-Needle2`.
