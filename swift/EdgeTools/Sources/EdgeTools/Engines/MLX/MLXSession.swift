#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && canImport(MLX)
  import _EdgeToolsFoundation
  import MLX
  import MLXLLM
  import MLXLMCommon
  import MLXNN
  import Observation

  #if canImport(CoreImage) && canImport(MLXVLM)
    import CoreImage
    import Foundation
    import MLXVLM
  #endif

  #if XGrammar
    extension EdgeToolsSession {
      public func clearCaches<Profile>() where Engine == MLXEngine<Profile>,
        Profile.GrammarEngine == XGrammarEngine
      {
        self.engine.clearCaches()
      }
    }
  #endif

  extension EdgeToolsSession {
    public func context<Profile>(
      transcript: EdgeToolsTranscript = EdgeToolsTranscript(),
      reasoningEffort: EdgeToolsReasoningEffort = .default
    ) -> MLXContext<Profile> where Engine == MLXEngine<Profile> {
      self.context(MLXContextParameters(transcript: transcript, reasoningEffort: reasoningEffort))
    }

    public func context<Profile>(
      systemPrompt: String,
      reasoningEffort: EdgeToolsReasoningEffort = .default
    ) -> MLXContext<Profile> where Engine == MLXEngine<Profile> {
      let transcript = EdgeToolsTranscript(messages: [.system(systemPrompt)])
      return self.context(transcript: transcript, reasoningEffort: reasoningEffort)
    }
  }

  private struct MLXTextVocabularyConfiguration: Decodable {
    var vocabularySize: Int

    enum CodingKeys: String, CodingKey {
      case vocabularySize = "vocab_size"
    }
  }

  private struct MLXVocabularyConfiguration: Decodable {
    var vocabularySize: Int?
    var textConfiguration: MLXTextVocabularyConfiguration?

    enum CodingKeys: String, CodingKey {
      case vocabularySize = "vocab_size"
      case textConfiguration = "text_config"
    }
  }

  private func mlxVocabularySize(from configurationData: Data) throws -> Int {
    let configuration = try JSONDecoder.json5()
      .decode(
        MLXVocabularyConfiguration.self,
        from: configurationData
      )
    guard
      let vocabularySize = configuration.vocabularySize
        ?? configuration.textConfiguration?.vocabularySize
    else {
      throw EdgeToolsError.failedToLoadConfiguration
    }
    return vocabularySize
  }

  private func mlxExtraStopTokenIds<Profile: MLXModelProfile>(
    profile: Profile.Type,
    directory: MLXModelDirectory,
    tokenizer: any EdgeToolsTokenizer
  ) throws -> Set<EdgeToolsToken.ID> {
    var tokenIds = try directory.loadStopTokenIds()
    tokenIds.formUnion(Profile.extraStopTokens.compactMap { tokenizer.convertTokenToId($0) })
    if let eosTokenId = tokenizer.eosTokenId { tokenIds.remove(eosTokenId) }
    return tokenIds
  }

  #if canImport(CoreImage) && canImport(MLXVLM) && HuggingFaceTokenizers && canImport(CTokenizers)
    private func mlxVLMProcessor(
      from directory: MLXModelDirectory,
      modelType: String,
      tokenizer: HuggingFaceTokenizer
    ) async throws -> sending any UserInputProcessor {
      let data = try directory.loadProcessorConfigurationData()
      let configuration = try JSONDecoder.json5()
        .decode(
          BaseProcessorConfiguration.self,
          from: data
        )
      let processorType =
        switch modelType {
        case "mistral3": "Mistral3Processor"
        case "gemma4_unified": "Gemma4UnifiedProcessor"
        default: configuration.processorClass
        }
      return try await VLMProcessorTypeRegistry.shared.createModel(
        configuration: data,
        processorType: processorType,
        tokenizer: tokenizer
      )
    }
  #endif

  private func loadMLXWeights(
    from directory: MLXModelDirectory,
    into model: sending any LanguageModel,
    configuration: BaseConfiguration,
    patchWeights: (
      _ weights: inout [String: MLXArray],
      _ model: any LanguageModel
    ) throws -> Void
  ) throws -> sending any LanguageModel {
    let safetensors = try directory.loadSafetensors()
    var weights = model.sanitize(
      weights: safetensors.weights,
      metadata: safetensors.mergedMetadata
    )
    try patchWeights(&weights, model)
    if let perLayerQuantization = configuration.perLayerQuantization {
      quantize(model: model) { path, _ in
        guard weights["\(path).scales"] != nil else { return nil }
        return perLayerQuantization.quantization(layer: path)?.asTuple
      }
    }
    try model.update(
      parameters: ModuleParameters.unflattened(weights),
      verify: [.all]
    )
    eval(model)
    return model
  }

  private func mlxPrefillContextMatches(
    cachedInput: LMInput?,
    input: LMInput,
    cachedContext: EdgeToolsLLMPrefillContext?,
    inputContext: EdgeToolsLLMPrefillContext?
  ) -> Bool {
    guard let cachedInput,
      mlxTextMasksHaveSamePrefix(cachedInput.text.mask, input.text.mask)
    else {
      return false
    }
    if let cachedContext, let inputContext {
      return cachedContext.hasMediaPrefix(in: inputContext)
    }
    return mlxProcessedImagesEqual(cachedInput.image, input.image)
      && mlxProcessedVideosEqual(cachedInput.video, input.video)
      && mlxProcessedAudioEqual(cachedInput.audio, input.audio)
  }

  private func mlxTextSuffix(_ text: LMInput.Text, from index: Int) -> LMInput.Text {
    let tokens =
      if text.tokens.ndim == 1 {
        text.tokens[index...][.newAxis]
      } else {
        text.tokens[.ellipsis, index...]
      }
    let mask = text.mask.map { mask in
      if mask.ndim == 1 {
        mask[index...][.newAxis]
      } else {
        mask[.ellipsis, index...]
      }
    }
    return LMInput.Text(tokens: tokens, mask: mask)
  }

  private func mlxTextMasksHaveSamePrefix(_ lhs: MLXArray?, _ rhs: MLXArray?) -> Bool {
    switch (lhs, rhs) {
    case (nil, nil):
      return true
    case (.some(let lhs), .some(let rhs)):
      guard lhs.ndim == rhs.ndim, lhs.shape.dropLast() == rhs.shape.dropLast(),
        lhs.dim(-1) <= rhs.dim(-1)
      else { return false }
      return mlxArraysEqual(lhs, rhs[.ellipsis, ..<lhs.dim(-1)])
    default:
      return false
    }
  }

  private func mlxProcessedImagesEqual(
    _ lhs: LMInput.ProcessedImage?,
    _ rhs: LMInput.ProcessedImage?
  ) -> Bool {
    switch (lhs, rhs) {
    case (nil, nil):
      true
    case (.some(let lhs), .some(let rhs)):
      mlxArraysEqual(lhs.pixels, rhs.pixels)
        && mlxArraysEqual(lhs.positionIds, rhs.positionIds)
        && mlxFramesEqual(lhs.frames, rhs.frames)
    default:
      false
    }
  }

  private func mlxProcessedVideosEqual(
    _ lhs: LMInput.ProcessedVideo?,
    _ rhs: LMInput.ProcessedVideo?
  ) -> Bool {
    switch (lhs, rhs) {
    case (nil, nil):
      true
    case (.some(let lhs), .some(let rhs)):
      mlxArraysEqual(lhs.pixels, rhs.pixels)
        && mlxArraysEqual(lhs.positionIds, rhs.positionIds)
        && mlxFramesEqual(lhs.frames, rhs.frames)
    default:
      false
    }
  }

  private func mlxProcessedAudioEqual(
    _ lhs: LMInput.ProcessedAudio?,
    _ rhs: LMInput.ProcessedAudio?
  ) -> Bool {
    switch (lhs, rhs) {
    case (nil, nil):
      true
    case (.some(let lhs), .some(let rhs)):
      mlxArraysEqual(lhs.features, rhs.features) && mlxArraysEqual(lhs.mask, rhs.mask)
    default:
      false
    }
  }

  private func mlxArraysEqual(_ lhs: MLXArray?, _ rhs: MLXArray?) -> Bool {
    switch (lhs, rhs) {
    case (nil, nil):
      true
    case (.some(let lhs), .some(let rhs)):
      lhs.dtype == rhs.dtype && lhs.shape == rhs.shape && lhs.arrayEqual(rhs).item(Bool.self)
    default:
      false
    }
  }

  private func mlxFramesEqual(_ lhs: [THW]?, _ rhs: [THW]?) -> Bool {
    switch (lhs, rhs) {
    case (nil, nil):
      true
    case (.some(let lhs), .some(let rhs)):
      lhs.count == rhs.count
        && zip(lhs, rhs)
          .allSatisfy { lhs, rhs in lhs.t == rhs.t && lhs.h == rhs.h && lhs.w == rhs.w }
    default:
      false
    }
  }
#endif
