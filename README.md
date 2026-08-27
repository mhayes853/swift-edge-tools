# Swift Edge Tools

> [!IMPORTANT] 
> This is still under (very early) active development. There will almost certainly be (many) breaking changes.

A flexible framework for edge inference.

```swift
import EdgeTools
import Foundation

struct GetWeather: EdgeTool {
  @EdgeToolsGenerable
  struct Input {
    @EdgeToolsGuide(.pattern("[A-Za-z0-9]+"))
    let city: String
  }

  let name = "get_weather"
  let description = "Fetches the weather for a city."

  func invoke(input: Input) async throws -> String {
    // Fetch the weather somehow...
  }
}

let engine = try await Gemma4MLXModelEngine(from: modelURL)
let session = EdgeToolsSession(engine: engine)

let context = session.context(
  MLXContextParameters(
    transcript: EdgeToolsTranscript(
      messages: [.system("You are an assistant who can fetch the weather.")]
    )
  )
) { 
  GetWeather() 
}
let response = try await session.respond(
  to: .user("What is the weather in San Francisco?"), 
  as: String.self,
  context: context
)
print(response.output)
```

## Overview

Edge inference can save costs, support privacy and offline modes, reduced latency, and handle many simple use-cases that most run on overpriced frontier models. Those are already known selling points, but none answer the question of how the end-user or developer experience improves as a result of being able to run inference on consumer hardware rather than in the cloud.

This framework is built out of the desire to explore what it means to create experiences that can _only_ run on the edge. Not just using the edge to act as a privacy, latency, or cost savings mechanism for frontier models. In other words, it's designed to be the engineering backbone of those experiences.

While the initial form of this framework takes a similar form to other LLM-based frameworks, this is not the intended end-destination, but merely a necessary starting point.

This README is primarily about the goals and direction the framework takes as the details haven't been hardened yet, and no documentation has been written. However, you can read [AGENTS.md](https://github.com/mhayes853/swift-edge-tools/blob/main/AGENTS.md) to understand the high-level details of how it works.

Right now, the framework supports dedicated MLX, llama, and [Needle 2](https://github.com/cactus-compute/needle) engines.

### Differences from Other Frameworks (eg. FoundationModels, Swarm, OpenAI Compatible Endpoints, etc.)

Most application frameworks are built solely around conversational or agentic LLMs. Furthermore, they often completely hide lower-level concerns (KV Cache policies/memory management, grammar constraints, etc.) from users in the name of providing a minimal API surface.

Both of these concerns are fine and necessary, but create rigid boundaries that are hard or outright impossible to work around once a use case no longer fits them. For example, [Needle 2](https://github.com/cactus-compute/needle) is a model that only emits tool calls and does not handle conversational settings very well, but it handles the tool calling case well. The lack of general conversation ability makes it a poor fit for existing frameworks, which assume all models conform to that structure.

Edge Tools still aims at providing a convenient interface to consume conversational LLMs like other frameworks do, but it also doesn't intend to end there. Even in its early form, you get access to models like Needle 2, some control over the KV Cache (forking), control over constrained generation through raw EBNF, etc. 

Furthermore, Edge Tools compiles for _every_ platform that Swift supports, including WASM and Embedded (Swift 6.4 only for now on embedded), and one of the only ones that does so to this date (though only Needle 2 works on WASM and Embedded for now). It even works without a Foundation import, which massively reduces the binary sizes for environments where that isn't ideal (mainly Android, WASM, and Embedded).

### Engines and Package Traits

The framework compiles quite lean without any traits enabled since it doesn't import Foundation, but if you want to do actual inference you'll need to enable the appropriate package traits to do so.
- `MLX` enables MLX inference.
- `Llama` enables llama inference.
  - There is also a dedicated `EdgeToolsLlama` target that wraps parts of the C API in Swift
- `XGrammar` enables XGrammar usage.
  - This is enabled by `MLX` and `Llama`. 
  - There is also a standalone `EdgeToolsXGrammar` target that wraps the entire XGrammar API in Swift.
  - XGrammar is depended on via a submodule, and this repo contains patches that allow it to compile in single-threaded WASI environments.
- `Needle2` enables support for the dedicated Needle 2 engine.
  - This is a wrapper around the official Needle2 binaries that are vendored in an artifactbundle to SPM.
  - Needle 2 can also been run in WASM environments through the typescript package in this repo, and `Needle2JSEngine`.
- `JS` enables JavaScriptKit support, which is required for `Needle2JSEngine`.
- `Foundation` to enable APIs that support Foundation (this is enabled by default).
- `FoundationEssentials` to enable APIs that need Foundation support, but not any of the internationalization.
  - This is enabled by `MLX` and `Llama`. 
- `ChatTemplates` enabled chat template rendering through [Minja](https://github.com/google/minja).
- `HuggingFaceTokenizers` integrates the framework with HF's tokenizers crate.
  - This is vendored as an artifact bundle over a minimal C interface to SPM.

There are hard dependencies on swift-collections and yyjson even with no traits enabled, but those are mostly required for dealing with parsing and formatting key-ordered JSON without Foundation. 

Eventually, [swift-stream-parsing](https://github.com/mhayes853/swift-stream-parsing) should supercede the need for those hard dependencies once it's [new architecture](https://github.com/mhayes853/swift-stream-parsing/pull/11) is merged. The new architecture can parse incomplete JSON values with throughputs at gigabytes per second (comparable to yyjson), and is more suited for parsing LLM JSON since the whole payload is streamed and parsed incrementally (yyjson requires the whole payload). This will also enable the same kind of typed partial streaming that FoundationModels offers, but at much faster speeds.

## Future Directions/Roadmap

The immediate goals over the coming weeks will be to get the shape of the initial framework right (mostly simplification), and more stable/battletested.

The overall goals of the framework are to make things possible that can _only_ be done on the edge. Often, this means multiple kinds of models working directly together (eg. STT transcribes audio into an LLM prompt for tool calls.), or other kinds of models that primarily do best in low-latency settings.

I would also like many of the existing dependencies (eg. Tokenizers, Chat Templating) to be ported in-house at some point, but this is not a priority for now.

Another direction is focusing downwards towards the engine/kernel layer. I think there are enough options that offer solid performance there, and what matters more for this framework is control. That includes being able to plug your own engine into the framework.

## Citations

Needle 2 and its weights are the work of [Cactus Compute](https://github.com/cactus-compute).

```bibtex
@misc{needle2_2026,
  title        = {Needle 2: A 45M-Parameter Foundation Tool-Calling Model for Tiny Devices},
  author       = {Ndubuaku, Henry and Mosoyan, Karen and Mroz, Jakub and Cylich, Noah and
                  Kumar, Satyajit and Sandhu, Parkirat and Shemet, Roman and Lee, Justin H.},
  year         = {2026},
  organization = {Cactus Compute, Inc.},
  howpublished = {\url{https://github.com/cactus-compute/needle}}
}
```
