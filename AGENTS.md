# Swift Edge Tools

You are working inside a Swift framework for deploying small on-device models that are capable of making tool calls. The framework is designed to run absolutely everywhere Swift supports, including WASM, Linux, Windows, Android, and even allocating-embedded Swift.

Our primary goal is to enable tiny models to run anywhere in the Swift ecosystem, and to be the most flexible framework for interacting with tool-calling capable language models. While Apple frameworks like FoundationModels support the E2E conversational use-case, the job of EdgeTools is to fit anywhere and be complementary to conversational frameworks like FoundationModels.

## Differences from other Frameworks

To understand the differences between this framework, and similar frameworks for working with LLMs in Swift, refer to this table.

| Category          | Edge Tools                                                   | Others (eg. FoundationModels, swift-transformers, etc.)      |
|-------------------|--------------------------------------------------------------|--------------------------------------------------------------|
| Model Support     | Primarily tiny on-device models (<1B parameters) with support for various architectures (Simple Attention Networks, Seq2Seq, Decoder-only, Encoder-only, etc) that are capable of making tool calls. | Any model capable of chatting, which includes frontier models. Primarily limited to autoregressive decoder-only models. |
| Engine Support    | On-device engines that run on consumer hardware or embedded devices (MLX, ONNX, CoreML, etc). May even support entirely-model specific engines depending on the use-case and whether or not the model warrants it. | Any engine that supports a chat interface. Often includes REST APIs over the network. |
| Platform Support  | Anywhere Swift runs, including WASM, Windows, Android, and allocating embedded environments. | Often Apple-platforms only, and maybe Linux and Android if lucky. |
| Design Philosophy | Drop in anywhere whether in a typical SwiftUI view/view model, backend endpoint, or even within other frameworks like FoundationModels. You pick and choose which parts of the framework to use depending on the scenario. | An all-consuming framework meant to handle the conversation workflow E2E. This makes it easy to use for application development at the expense of creating a black box. |
| Optimizations     | Certain models that fit the ideals of the framework (tiny, tool-calling capable) best will often have dedicated optimzations. This requires more work/complexity, but leads to better overall performance. | All models are treated equal. Less model-specific complexity, but comes at the cost of obtaining maximum performance. |

## Files + Layout

The framework is organized into a `swift` and `python` directories.

The `swift` side contains the actual Swift framework, as well as other products that can be consumed by end-users.

The `python` side contains python code for handling modeling and exporting of various models that the framework optimizes for. It is generally split into sub-directories by model-type with a single file to run a basic CLI at the root.

The `swift` side also may contain patch files that are applied through a custom build plugin. Feel free to add patches to any third-party libraries provided they aren't excessive changes.

## Framework Basics

On the Swift side, `EdgeToolsSession` is currently the primary way to interact with a model. It manages active generation streams on the model, but unlike other frameworks does not hold onto conversation history (the user passes that through the session). It conforms to `Observable`, and so does the streams it manages.

The session also consumes a generic `EdgeToolsEngine`, which is the underlying protocol for handling generation. Most engine conformances share the same logic for the overall generation loop and grammar constraints, and build on top of `EdgeToolsModelActorEngine`. However, other future engines may implement the generation loop themselves, in which case it would be more correct to conform to `EdgeToolsEngine` directly.

Engines generally handle the following responsibilities:

- Grammar constraints (typically using XGrammar).
- Tool call parsing.
  - Tool call parsing is done incrementally, meaning that the moment enough tokens have been emitted to parse the information for a single tool call, we immediately publish it through the generation channel. This allows decoding to continue while the tool call is invoked by the session in the background.
- The full generation loop, including the ability to control when it starts and stops.
- An `Model` protocol that adapts specific models to the engine.
  - Certain engines may want to have general implementations for this protocol for LLM-based models that support simple conversational workflows (eg. Qwen, Gemma, etc). This is because those models generally function the same architecturally.

A `Model` protocol generally consists of:

- A model-specific tool call grammar.
- A model-specific tool call parser.
- A preparation step that runs before generation.
  - This is not a prefill, but rather something that runs right before the decode phase. This is because not all supported models can have conventional prefill phases.
  - For models that can support conventional prefill phases, an engine should offer a `PrefillableModel` protocol with a `prefill` requirement. The `prefill` requirement can be used as the default implementation for `prepare`.
- A decode step to predict the next token.
  - This step includes logit sampling and masking.

The session represents generally collects tool calls into an `EdgeToolsToolCallCollection`. This collection contains individual `AnyEdgeToolCall` instances that represent the active response state of the tool call, and this can be further casted down a strongly typed `EdgeToolCall`. One must `await` the `output` of a tool call since the tool may or may not be actively responding. Multiple calls to `output` are deduplicated.

Grammar constraints and tool argument parsing are generally represented through the `EdgeToolsGenerationSchema` struct and associated conversion protocols. The schema struct represents a valid JSON schema object under the hood. Generally, a user doesn't need to create generation schema's by hand because the `@EdgeToolsGenerable` macro can be applied to any struct. This macro will create a schema for the type, and conform the type to various protocols that use the schema.

## Code Conventions

Generally speaking, follow the patterns you see already for the most part. The following are generally mistakes that previous agents have made.

One-line `if` statements with a `return` on the same line are not allowed. Non-empty function signatures and bodies must be on separate lines. Deinits and empty functions may remain on one line.

When a protocol requirement has an unused parameter, preserve its existing parameter name rather than expressing it with an underscore binding such as `prompt _:`. Use `prompt:` instead.

Avoid using things like `.init` or making excessive amounts of static functions. If a function doesn't reasonably belong on a type, feel free to make it a global function.

Separate significant sections of functionallity or types with MARK comments. MARK comments should never appear inside type, extension, or function bodies. Always keep them at the file scope.

Try not to be overly verbose, focus on making things condensed.

Try to prioritize using Collections/Sequence algorithms over doing things with loops as long as it doesn't look too crazy.

Private helper functions should go at the bottom of the file, not the top.

DO NOT ASSUME WASI IS A SINGLE THREADED ENVIRONMENT. It is not ok to conform types to `@unchecked Sendable` just because they have a member variable that uses a type from `JavaScriptKit`, and because "JS is single-threaded". `JavaScriptKit` does not conform most of its types to Sendable because Swift WASM supports sdks that enable multithreading through web workers or other means.

If a lot of public APIs are using shared package or internal scoped APIs, it's likely that the API should also be public since it has inherent useful reusability. That is, users should get the same tools as us to make their own abstractions.

## Testing

Generally, focus tests only on the public API (try to avoid anything not marked as public), and follow the conventions in the existing test suite. The following are generally mistakes that previous agents have made.

Do not test obvious functionallity like "member-wise initializers init properly" or "synthesized codable works properly". Avoid explicitly testing anything that is tautological (ie. where the assertion is close to the implementation code).

Test suite names must use the `PascalCase tests` convention, with no spaces within the PascalCase portion (for example, `MySuite tests`). Do not use "Consolidated" in test suite names.

The highest levels of the framework (eg. `EdgeToolsSession`, etc.) should be tested in a manner that is not tied to any specific engine or model. This ensures the framework itself works no matter what engine is powering the session.

The most important tests besides the general framework tests are the engine generation tests and the snapshots they produce. Everything else is trivial by comparison. Most engine generation tests record snapshots with `withKnownIssue`, this makes them robust to non-deterministic model outputs, but requires manual validation of the snapshot. As an agent, you should be able to perform such validation yourself.

Run MLX tests through `scripts/test-mlx.sh`. The script enables the required
traits and configures MLX's Metal library for the SwiftPM test runner. Use
`scripts/test-mlx.sh --filter '<test filter>'` to run a focused subset.

Make sure to prefer using `expectNoDifference` over `#expect` for assertions. The only times where `#expect` are preferred are for `#expect(throws:)` and for `WASITests` (`expectNoDifference` doesn't work properly on WASI). For assertions on boolean expressions, you can do `expectNoDifference(myConditionExpression, true)` or `expectNoDifference(myConditionExpression, false)`.

When snapshots update from new generation tests, don't delete or revert them. Leave them as is.

`test_wasm.sh` and `test_linux.sh` can be used to run WASM and Linux tests on macOS.
