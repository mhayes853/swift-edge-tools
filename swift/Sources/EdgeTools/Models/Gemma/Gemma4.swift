#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && XGrammar && Transformers && canImport(MLX) && canImport(MLXVLM)
  import Foundation
  import MLX
  import MLXHuggingFace
  import MLXLMCommon
  import MLXNN
  import MLXVLM
  import Tokenizers

  // MARK: - Gemma 4 Model

  public struct Gemma4MLXModel: MLXModel {
    public struct Configuration: Sendable {
      public var model: MLXVLM.Gemma4Configuration
      public var processor: MLXVLM.Gemma4ProcessorConfiguration

      public init(
        model: MLXVLM.Gemma4Configuration,
        processor: MLXVLM.Gemma4ProcessorConfiguration
      ) {
        self.model = model
        self.processor = processor
      }
    }

    public typealias ModelConfiguration = Configuration
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = Gemma4ToolCallParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public let languageModel: MLXVLM.Gemma4
    private let processorConfiguration: MLXVLM.Gemma4ProcessorConfiguration

    public init(configuration: Configuration) {
      self.languageModel = MLXVLM.Gemma4(configuration.model)
      self.processorConfiguration = configuration.processor
    }

    public var vocabularySize: Int { self.languageModel.vocabularySize }
    public var extraStopTokens: Set<String> { ["<|tool_response>"] }

    public func loadWeights(from directoryURL: URL) throws {
      guard
        let baseConfiguration = try decodeModelConfiguration(
          BaseConfiguration.self,
          in: directoryURL,
          decoder: JSONDecoder.json5()
        )
      else {
        throw EdgeToolsError.failedToLoadConfiguration
      }

      var (weights, metadata) = try loadGemma4Weights(from: directoryURL)
      weights = self.languageModel.sanitize(weights: weights, metadata: metadata)

      // NB: mlx-swift-lm 3.31.4's VLM backbone declares local K/V projections for the shared-KV
      // tail even though E2B/E4B checkpoints correctly omit them.
      let firstSharedLayer =
        self.languageModel.config.textConfiguration.hiddenLayers
        - self.languageModel.config.textConfiguration.numKVSharedLayers
      if firstSharedLayer > 0 {
        for (key, value) in self.languageModel.parameters().flattened() where weights[key] == nil {
          let components = key.split(separator: ".")
          guard
            components.count > 6,
            components[0] == "language_model",
            components[1] == "model",
            components[2] == "layers",
            let layer = Int(components[3]),
            layer >= firstSharedLayer,
            components[4] == "self_attn",
            components[5] == "k_proj" || components[5] == "v_proj"
              || components[5] == "k_norm"
          else { continue }
          weights[key] = value
        }
      }

      if let perLayerQuantization = baseConfiguration.perLayerQuantization {
        quantize(model: self.languageModel) { path, _ in
          guard weights["\(path).scales"] != nil else { return nil }
          return perLayerQuantization.quantization(layer: path)?.asTuple
        }
      }
      try self.languageModel.update(
        parameters: ModuleParameters.unflattened(weights),
        verify: [.all]
      )
      eval(self.languageModel)
    }

    public func toolCallGrammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try .gemma4(tools: tools, range: range)
    }

    public nonisolated(nonsending) func input(
      prompt: EdgeToolsLLMPrompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer
    ) async throws -> EdgeToolsModelInput<LMInput> {
      guard let tokenizer = tokenizer as? TransformersTokenizer else {
        throw EdgeToolsError.unsupportedTokenizer
      }
      let processor = MLXVLM.Gemma4Processor(
        self.processorConfiguration,
        tokenizer: #adaptHuggingFaceTokenizer(tokenizer.base)
      )
      let input = try await processor.prepare(input: try prompt.gemma4UserInput(tools: tools))
      let tokenIds = input.text.tokens.asArray(EdgeToolsToken.ID.self)
      return EdgeToolsModelInput(value: input, tokenIds: tokenIds)
    }
  }

  public typealias Gemma4Configuration = MLXVLM.Gemma4Configuration
  public typealias Gemma4MLXModelEngine = MLXEngine<Gemma4MLXModel>

  extension Gemma4MLXModelEngine {
    public init(from directoryURL: URL) async throws {
      guard
        let modelConfiguration = try decodeModelConfiguration(
          MLXVLM.Gemma4Configuration.self,
          in: directoryURL,
          decoder: JSONDecoder.json5()
        ),
        let processorConfiguration = try decodeMLXVLMProcessorConfiguration(
          MLXVLM.Gemma4ProcessorConfiguration.self,
          in: directoryURL
        )
      else {
        throw EdgeToolsError.failedToLoadConfiguration
      }

      try await self.init(
        from: directoryURL,
        configuration: Gemma4MLXModel.Configuration(
          model: modelConfiguration,
          processor: processorConfiguration
        )
      )
    }
  }

  extension EdgeToolsLLMPrompt {
    fileprivate func gemma4UserInput(tools: [EdgeToolDefinition]) throws -> UserInput {
      try self.mlxVLMUserInput(tools: tools) { message in
        switch message {
        case .system:
          return try message.mlxMessage()
        case .user(let text, let messageImages, audio: _):
          var content: [MLXLMCommon.Message] = messageImages.map { _ in ["type": "image"] }
          content.append(["type": "text", "text": text])
          return ["role": "user", "content": content]
        case .assistant, .tool:
          var result = try message.mlxMessage()
          if let text = result["content"] as? String {
            result["content"] = [["type": "text", "text": text]] as [MLXLMCommon.Message]
          }
          return result
        }
      }
    }
  }

  private func loadGemma4Weights(
    from directoryURL: URL
  ) throws -> (weights: [String: MLXArray], metadata: [String: String]) {
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    let enumerator = FileManager.default.enumerator(
      at: directoryURL,
      includingPropertiesForKeys: nil
    )!
    for case let url as URL in enumerator where url.pathExtension == "safetensors" {
      let (arrays, fileMetadata) = try loadArraysAndMetadata(url: url)
      weights.merge(arrays) { _, new in new }
      if metadata.isEmpty { metadata = fileMetadata }
    }
    return (weights, metadata)
  }
#endif
