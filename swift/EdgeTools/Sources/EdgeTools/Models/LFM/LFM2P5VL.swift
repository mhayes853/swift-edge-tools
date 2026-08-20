#if MLX && XGrammar && canImport(CoreImage) && canImport(MLX) && canImport(MLXVLM)
  import EdgeToolsCore
  import EdgeToolsTokenizers
  import Foundation
  import MLXLMCommon
  import MLXVLM
  import EdgeToolsXGrammar

  // MARK: - LFM2P5VL Model

  public struct LFM2P5VLMLXProfile:
    MLXVLMModelProfile,
    EdgeToolsMultimodalModelProfile {
    public typealias Prompt = EdgeToolsTranscript
    public typealias GenerationParser = LFM2P5GenerationParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarEngine = XGrammarEngine

    public static func multimodalContent(
      for message: EdgeToolsTranscript.UserMessage
    ) -> [EdgeToolsMultimodalContent] {
      [.text(message.content)] + message.images.map(EdgeToolsMultimodalContent.image)
    }

    public static func grammar(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      parameters: DefaultMLXGenerateParameters,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(tools: tools, parameters: parameters, grammarEngine: grammarEngine) {
        try .lfm2P5(tools: tools, range: $0)
      }
    }

    public static nonisolated(nonsending) func input(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput {
      guard let processor else { throw EdgeToolsError.failedToLoadConfiguration }
      return try await processor.prepare(
        input: try prompt.lfm2P5VLUserInput(tools: tools, addGenerationPrompt: true)
      )
    }

    public static nonisolated(nonsending) func prefillInput(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput {
      guard let processor else { throw EdgeToolsError.failedToLoadConfiguration }
      return try await processor.prepare(
        input: try prompt.lfm2P5VLUserInput(tools: tools, addGenerationPrompt: false)
      )
    }
  }

  public typealias LFM2P5VLMLXModelEngine = MLXEngine<LFM2P5VLMLXProfile>

  #if HuggingFaceTokenizers && canImport(CTokenizers)
    extension MLXEngine where Profile == LFM2P5VLMLXProfile {
      public convenience init(from directoryURL: URL) async throws {
        try await self.init(from: MLXModelDirectory(url: directoryURL))
      }

      public convenience init(from directory: MLXModelDirectory) async throws {
        try await self.init(from: directory) { weights, model in
          guard let model = model as? MLXVLM.LFM2VL, !model.config.projectorUseLayernorm
          else { return }
          let staleKeys = weights.keys.filter {
            $0.hasPrefix("multi_modal_projector.layer_norm.")
          }
          for key in staleKeys {
            weights.removeValue(forKey: key)
          }
        }
      }
    }
  #endif

  extension EdgeToolsTranscript {
    fileprivate func lfm2P5VLUserInput(
      tools: [EdgeToolDefinition],
      addGenerationPrompt: Bool
    ) throws -> UserInput {
      try self.mlxUserInput(
        tools: tools,
        additionalContext: ["add_generation_prompt": .boolean(addGenerationPrompt)]
      ) { message in
        guard case .user(let message) = message else {
          return try message.mlxMessage()
        }
        guard !message.images.isEmpty else { return ["role": "user", "content": message.content] }
        return [
          "role": "user",
          "content": LFM2P5VLMLXProfile.multimodalContent(for: message).map(\.mlxMessage)
        ]
      }
    }
  }
#endif

#if Llama && XGrammar && canImport(CLlama)
  import EdgeToolsXGrammar

  // MARK: - LFM2P5VL Llama Model

  public struct LFM2P5VLLlamaProfile:
    LlamaModelProfile,
    EdgeToolsMultimodalModelProfile {
    public typealias Prompt = EdgeToolsTranscript
    public typealias GenerationParser = LFM2P5GenerationParser
    public typealias GenerateParameters = DefaultLlamaGenerateParameters
    public typealias GrammarEngine = XGrammarEngine

    public static func multimodalContent(
      for message: EdgeToolsTranscript.UserMessage
    ) -> [EdgeToolsMultimodalContent] {
      [.text(message.content)] + message.images.map(EdgeToolsMultimodalContent.image)
    }

    public static func grammar(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      parameters: DefaultLlamaGenerateParameters,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(tools: tools, parameters: parameters, grammarEngine: grammarEngine) {
        try .lfm2P5(tools: tools, range: $0)
      }
    }
  }

  public typealias LFM2P5VLLlamaModelEngine = LlamaEngine<LFM2P5VLLlamaProfile>
#endif
