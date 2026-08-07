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

  // MARK: - LFM2.5-VL Model

  public struct LFM2P5VLMLXModel: MLXModel {
    public struct Configuration: Sendable {
      public var model: MLXVLM.LFM2VLConfiguration
      public var processor: MLXVLM.LFM2VLProcessorConfiguration

      public init(
        model: MLXVLM.LFM2VLConfiguration,
        processor: MLXVLM.LFM2VLProcessorConfiguration
      ) {
        self.model = model
        self.processor = processor
      }
    }

    public typealias ModelConfiguration = Configuration
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = LFM2P5VLToolCallParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public let languageModel: MLXVLM.LFM2VL
    private let processorConfiguration: MLXVLM.LFM2VLProcessorConfiguration

    public init(configuration: Configuration) {
      self.languageModel = MLXVLM.LFM2VL(configuration.model)
      self.processorConfiguration = configuration.processor
    }

    public var vocabularySize: Int { self.languageModel.vocabularySize }

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

      var (weights, metadata) = try loadLFM2P5VLWeights(from: directoryURL)
      weights = self.languageModel.sanitize(weights: weights, metadata: metadata)

      // NB: The 450M MLX checkpoint disables projector LayerNorm in config but retains weights from
      // an older conversion that always created it. The current module correctly has no matching
      // child; Python mlx-vlm drops these stale tensors before strict loading, so we do the same.
      if !self.languageModel.config.projectorUseLayernorm {
        let staleKeys = weights.keys.filter {
          $0.hasPrefix("multi_modal_projector.layer_norm.")
        }
        for key in staleKeys {
          weights.removeValue(forKey: key)
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
      try .lfm2(tools: tools, range: range)
    }

    public nonisolated(nonsending) func input(
      prompt: EdgeToolsLLMPrompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer
    ) async throws -> EdgeToolsModelInput<LMInput> {
      guard let tokenizer = tokenizer as? TransformersTokenizer else {
        throw EdgeToolsError.unsupportedTokenizer
      }
      let processor = MLXVLM.LFM2VLProcessor(
        self.processorConfiguration,
        tokenizer: #adaptHuggingFaceTokenizer(tokenizer.base)
      )
      let input = try await processor.prepare(input: try prompt.lfm2VLUserInput(tools: tools))
      let tokenIds = input.text.tokens.asArray(EdgeToolsToken.ID.self)
      return EdgeToolsModelInput(value: input, tokenIds: tokenIds)
    }
  }

  public typealias LFM2P5VLConfiguration = MLXVLM.LFM2VLConfiguration
  public typealias LFM2P5VLMLXModelEngine = MLXEngine<LFM2P5VLMLXModel>

  extension LFM2P5VLMLXModelEngine {
    public init(from directoryURL: URL) async throws {
      guard
        let modelConfiguration = try decodeModelConfiguration(
          MLXVLM.LFM2VLConfiguration.self,
          in: directoryURL,
          decoder: JSONDecoder.json5()
        ),
        let processorConfiguration = try decodeMLXVLMProcessorConfiguration(
          MLXVLM.LFM2VLProcessorConfiguration.self,
          in: directoryURL
        )
      else {
        throw EdgeToolsError.failedToLoadConfiguration
      }

      try await self.init(
        from: directoryURL,
        configuration: LFM2P5VLMLXModel.Configuration(
          model: modelConfiguration,
          processor: processorConfiguration
        )
      )
    }
  }

  extension EdgeToolsLLMPrompt {
    fileprivate func lfm2VLUserInput(tools: [EdgeToolDefinition]) throws -> UserInput {
      try self.mlxVLMUserInput(tools: tools) { message in
        guard case .user(let text, let messageImages, audio: _) = message else {
          return try message.mlxMessage()
        }
        guard !messageImages.isEmpty else { return ["role": "user", "content": text] }

        var content: [MLXLMCommon.Message] = [["type": "text", "text": text]]
        content.append(contentsOf: messageImages.map { _ in ["type": "image"] })
        return ["role": "user", "content": content]
      }
    }
  }

  private func loadLFM2P5VLWeights(
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

// MARK: - LFM2.5-VL Tool Calling

public typealias LFM2P5VLToolCallParser = LFM2PythonToolCallParser
