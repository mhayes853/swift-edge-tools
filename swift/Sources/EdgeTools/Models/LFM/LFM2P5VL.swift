#if MLX && XGrammar && canImport(CoreImage) && canImport(MLX) && canImport(MLXVLM)
  import Foundation
  import MLXLMCommon
  import MLXVLM
  import EdgeToolsXGrammar

  // MARK: - LFM2P5VL Model

  public struct LFM2P5VLMLXProfile: MLXVLMModelProfile {
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias GenerationParser = LFM2P5GenerationParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public static func grammar(
      prompt _: EdgeToolsLLMPrompt,
      tools: [EdgeToolDefinition],
      parameters: DefaultMLXGenerateParameters,
      context: XGRGrammarContext
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(tools: tools, parameters: parameters, context: context) {
        try XGRGrammar.lfm2P5(tools: tools, range: $0)
      }
    }

    public static nonisolated(nonsending) func input(
      prompt: EdgeToolsLLMPrompt,
      tools: [EdgeToolDefinition],
      tokenizer _: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput {
      guard let processor else { throw EdgeToolsError.failedToLoadConfiguration }
      return try await processor.prepare(input: try prompt.lfm2P5VLUserInput(tools: tools))
    }
  }

  public typealias LFM2P5VLMLXModelEngine = MLXEngine<LFM2P5VLMLXProfile>

  #if canImport(Tokenizers)
    extension EdgeToolsModelEngine where Model == EdgeToolsMLXModel<LFM2P5VLMLXProfile> {
      public init(from directoryURL: URL) async throws {
        try await self.init(from: MLXModelDirectory(url: directoryURL))
      }

      public init(from directory: MLXModelDirectory) async throws {
        try await self.init(from: directory) { weights, model in
          guard let model = model as? MLXVLM.LFM2VL,
            !model.config.projectorUseLayernorm
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

  extension EdgeToolsLLMPrompt {
    fileprivate func lfm2P5VLUserInput(tools: [EdgeToolDefinition]) throws -> UserInput {
      try self.mlxUserInput(tools: tools) { message in
        guard case .user(let text, let messageImages, videos: _, audio: _) = message else {
          return try message.mlxMessage()
        }
        guard !messageImages.isEmpty else { return ["role": "user", "content": text] }

        var content: [MLXLMCommon.Message] = [["type": "text", "text": text]]
        content.append(contentsOf: messageImages.map { _ in ["type": "image"] })
        return ["role": "user", "content": content]
      }
    }
  }

#endif
