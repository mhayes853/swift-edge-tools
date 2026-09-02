#if XGrammar
  import EdgeToolsCore
  import EdgeToolsXGrammar

  // MARK: - LFM2P5VL Model

  public struct LFM2P5VLProfile: EdgeToolsMultimodalModelProfile {
    public typealias GenerationParser = LFM2P5GenerationParser
    public typealias GrammarEngine = XGrammarEngine
    public typealias Constraint = XGRGenerationConstraint

    public static func multimodalContent(
      for message: EdgeToolsTranscript.UserMessage
    ) -> [EdgeToolsMultimodalContent] {
      [.text(message.content)] + message.images.map(EdgeToolsMultimodalContent.image)
    }

    public static func grammar(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      constraint: XGRGenerationConstraint,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(
        tools: tools,
        constraint: constraint,
        grammarEngine: grammarEngine
      ) {
        try .lfm2P5(tools: tools, range: $0)
      }
    }
  }
#endif

#if MLX && canImport(CoreImage) && canImport(MLX) && canImport(MLXVLM)
  import EdgeToolsTokenizers
  import Foundation
  import MLXLMCommon
  import MLXVLM

  // MARK: - LFM2P5VL MLX Model

  extension LFM2P5VLProfile: MLXVLMModelProfile {
    public static nonisolated(nonsending) func input(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
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
      reasoningEffort: EdgeToolsReasoningEffort,
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

  public typealias LFM2P5VLMLXModelEngine = MLXEngine<LFM2P5VLProfile>

  #if HuggingFaceTokenizers && canImport(CTokenizers)
    extension MLXEngine where Profile == LFM2P5VLProfile {
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
#endif

#if Llama && canImport(CLlama)
  // MARK: - LFM2P5VL Llama Model

  extension LFM2P5VLProfile: LlamaModelProfile {}

  public typealias LFM2P5VLLlamaModelEngine = LlamaEngine<LFM2P5VLProfile>
#endif

// MARK: - LFM2P5VL MLX Input

#if MLX && canImport(CoreImage) && canImport(MLX) && canImport(MLXVLM)
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
          "content": LFM2P5VLProfile.multimodalContent(for: message).map(\.mlxMessage)
        ]
      }
    }
  }
#endif
