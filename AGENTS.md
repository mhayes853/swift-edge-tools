# Swift Edge Tools
You are working inside a Swift framework for deploying small on-device models that are capable of making tool calls. The framework is designed to run absolutely everywhere Swift supports, including WASM, Linux, Windows, Android, and even allocating-embedded Swift. 

Our primary goal is to enable tiny models to run anywhere in the Swift ecosystem, and to be the most flexible framework for interacting with tool-calling capable language models. While Apple frameworks like FoundationModels support the E2E conversational use-case, the job of EdgeTools is to fit anywhere, and be complementary to conversational frameworks like FoundationModels.

## Differences from other Frameworks
To understand the differences between this framework, and similar frameworks for working with LLMs in Swift, refer to this table.
| Category          | Edge Tools                                                   | Others (eg. FoundationModels, swift-transformers, etc.)      |
|-------------------|--------------------------------------------------------------|--------------------------------------------------------------|
| Model Support     | Primarily tiny on-device models (<1B parameters) with support for various architectures (Simple Attention Networks, Seq2Seq, Decoder-only, Encoder-only, etc) that are capable of making tool calls. | Any model capable of chatting, which includes frontier models. Primarily limited to autoregressive decoder-only models. |
| Engine Support    | On-device engines that run on consumer hardware or embedded devices (MLX, ONNX, CoreML, etc). May even support entirely-model specific engines depending on the use-case and whether or not the model warrants it. | Any engine that supports a chat interface. Often includes REST APIs over the network. |
| Platform Support  | Anywhere Swift runs, including WASM, Windows, Android, and allocating embedded environments. | Often Apple-platforms only, and maybe Linux and Android if lucky. |
| Design Philosophy | Drop in anywhere whether in a typical SwiftUI view/view model, backend endpoint, or even within other frameworks like FoundationModels. You pick and choose which parts of the framework to use depending on the scenario. | An all-consuming framework meant to handle the conversation workflow E2E. This makes it easy to use for application development at the expense of creating a black box. |
| Optimizations     | Certain models that fit the ideals of the framework (tiny, tool-calling capable) best will often have dedicated optimzations. This requires more work/complexity, but leads to better overall performance. | All models are treated equal. Less model-specific complexity, but comes at the cost of obtaining maximum performance. |
