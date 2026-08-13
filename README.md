# Swift Edge Tools

A Swift framework for running small local models that can generate structured tool calls.

## The `edge` CLI

```sh
source ./setup.sh

# A Hugging Face repository, cached under ${HF_HOME:-~/.cache/huggingface}
edge Qwen/Qwen3-0.6B -p "Set a timer for 20 minutes" --tools my_tools.json

# A local model directory
edge --path ./models/qwen3 -p "..." --tools my_tools.json

# Inspect detected model support and available engines
edge info Qwen/Qwen3-0.6B

# Benchmark repeated generations
edge bench Qwen/Qwen3-0.6B -p "..." --repeat-count 20 --warmup 3 --json
```

The CLI detects models from `config.json` and selects an available registered engine from
the model weights. MLX uses `.safetensors` weights and is preferred on Apple platforms.
The `onnx` engine option remains available for models that register ONNX support.

Text models include Qwen3, Qwen3.5, LFM2, FunctionGemma, Granite, Granite MoE Hybrid, and
MiniCPM5. Qwen3.5 VL, Gemma4, and LFM2.5 VL are detected as vision models. Other supported
architectures fall back to generic text or vision MLX profiles.

## Package traits

`Foundation` is enabled by default. `MLX`, `Transformers`, and `FoundationModels` enable it
automatically. `XGrammar` provides structured generation, while `ONNXCore` and `ONNX` provide
generic ONNX runtime and provider APIs for models that use them.

```sh
swift build --disable-default-traits --traits Foundation
```

## Using a model engine

```swift
import EdgeTools

let engine = try await Qwen3MLXModelEngine(from: modelURL)
let session = EdgeToolsSession(engine: engine, tools: [GetWeather()])

let generation = try await session.generate(
  prompt: EdgeToolsTranscript(messages: [.user(.init(content: "What is the weather in San Francisco?"))]),
  context: nil,
  parameters: .default
)

print(generation.response)
```

Define tools with `EdgeTool` and use `@EdgeToolsGenerable` for structured inputs.
