#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && canImport(MLX)
  import EdgeToolsCore
  import EdgeToolsTokenizers
  import OrderedCollections
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

  extension MLXEngine {
    public convenience init(
      from directoryURL: URL,
      patchWeights: (
        _ weights: inout [String: MLXArray],
        _ model: any LanguageModel
      ) throws -> Void = { _, _ in }
    ) async throws where Profile: MLXLLMModelProfile {
      try await self.init(
        from: MLXModelDirectory(url: directoryURL),
        patchWeights: patchWeights
      )
    }

    public convenience init(
      from directory: MLXModelDirectory,
      patchWeights: (
        _ weights: inout [String: MLXArray],
        _ model: any LanguageModel
      ) throws -> Void = { _, _ in }
    ) async throws where Profile: MLXLLMModelProfile {
      let configurationData = try directory.loadConfigurationData()
      let baseConfiguration = try JSONDecoder.json5()
        .decode(BaseConfiguration.self, from: configurationData)
      let tokenizer = try await directory.loadTokenizer()
      let languageModel = try await LLMTypeRegistry.shared.createModel(
        configuration: configurationData,
        modelType: baseConfiguration.modelType
      )
      try self.init(
        from: directory,
        configurationData: configurationData,
        baseConfiguration: baseConfiguration,
        languageModel: languageModel,
        tokenizer: tokenizer,
        processor: nil,
        patchWeights: patchWeights
      )
    }

    #if canImport(CoreImage) && canImport(MLXVLM) && HuggingFaceTokenizers && canImport(CTokenizers)
      public convenience init(
        from directoryURL: URL,
        patchWeights: (
          _ weights: inout [String: MLXArray],
          _ model: any LanguageModel
        ) throws -> Void = { _, _ in }
      ) async throws where Profile: MLXVLMModelProfile {
        try await self.init(
          from: MLXModelDirectory(url: directoryURL),
          patchWeights: patchWeights
        )
      }

      public convenience init(
        from directory: MLXModelDirectory,
        patchWeights: (
          _ weights: inout [String: MLXArray],
          _ model: any LanguageModel
        ) throws -> Void = { _, _ in }
      ) async throws where Profile: MLXVLMModelProfile {
        let configurationData = try directory.loadConfigurationData()
        let baseConfiguration = try JSONDecoder.json5()
          .decode(BaseConfiguration.self, from: configurationData)
        let tokenizer = try await directory.loadTokenizer()
        guard let tokenizer = tokenizer as? HuggingFaceTokenizer else {
          throw EdgeToolsError.unsupportedTokenizer
        }
        let languageModel = try await VLMTypeRegistry.shared.createModel(
          configuration: configurationData,
          modelType: baseConfiguration.modelType
        )
        let processor = try await mlxVLMProcessor(
          from: directory,
          modelType: baseConfiguration.modelType,
          tokenizer: tokenizer
        )
        try self.init(
          from: directory,
          configurationData: configurationData,
          baseConfiguration: baseConfiguration,
          languageModel: languageModel,
          tokenizer: tokenizer,
          processor: processor,
          patchWeights: patchWeights
        )
      }
    #endif

    private convenience init(
      from directory: MLXModelDirectory,
      configurationData: Data,
      baseConfiguration: BaseConfiguration,
      languageModel: sending any LanguageModel,
      tokenizer: sending any EdgeToolsTokenizer,
      processor: sending (any UserInputProcessor)?,
      patchWeights: (
        _ weights: inout [String: MLXArray],
        _ model: any LanguageModel
      ) throws -> Void
    ) throws {
      let loadedLanguageModel = try loadMLXWeights(
        from: directory,
        into: languageModel,
        configuration: baseConfiguration,
        patchWeights: patchWeights
      )
      let vocabularySize = try mlxVocabularySize(from: configurationData)
      let extraStopTokenIds = try mlxExtraStopTokenIds(
        profile: Profile.self,
        directory: directory,
        tokenizer: tokenizer
      )
      try self.init(
        languageModel: loadedLanguageModel,
        tokenizer: tokenizer,
        processor: processor,
        vocabularySize: vocabularySize,
        extraStopTokenIds: extraStopTokenIds,
        defaultSampling: try? directory.loadDefaultSampling()
      )
    }
  }

  // MARK: - MLXEngine

  public final class MLXEngine<Profile: MLXModelProfile>:
    EdgeToolsEngine, EdgeToolsPrefillableEngine
  where Profile.Prompt == EdgeToolsTranscript {
    public typealias Context = MLXContext<Profile>
    public typealias ContextParameters = MLXContextParameters
    public typealias Prompt = EdgeToolsTranscript.Prompt
    public typealias GenerateParameters = Profile.GenerateParameters
    public typealias GenerationParser = Profile.GenerationParser
    public typealias GrammarEngine = Profile.GrammarEngine

    private let runtime: MLXRuntime<Profile>
    private let inputProcessor: MLXInputProcessor<Profile>
    private let configuredSampling: EdgeToolsFusedSamplingParameters?
    private let identity = EdgeToolsEngineIdentity()
    private let generationLoop: EdgeToolsGenerationLoop
    public let tokenizer: any EdgeToolsTokenizer
    public let grammarEngine: Profile.GrammarEngine

    public init(
      languageModel: sending any LanguageModel,
      tokenizer: sending any EdgeToolsTokenizer,
      processor: sending (any UserInputProcessor)? = nil,
      vocabularySize: Int,
      extraStopTokenIds: Set<EdgeToolsToken.ID> = [],
      defaultSampling: EdgeToolsFusedSamplingParameters? = nil
    ) throws {
      var stopTokenIds = extraStopTokenIds
      if let eosTokenId = tokenizer.eos?.id {
        stopTokenIds.insert(eosTokenId)
      }
      self.grammarEngine = try Profile.grammarEngine(
        tokenizer: tokenizer,
        vocabularySize: vocabularySize,
        stopTokenIds: stopTokenIds
      )
      let inputProcessor = MLXInputProcessor<Profile>(processor: processor, tokenizer: tokenizer)
      self.inputProcessor = inputProcessor
      self.runtime = MLXRuntime(
        languageModel: languageModel,
        inputProcessor: inputProcessor
      )
      self.configuredSampling = defaultSampling
      self.generationLoop = EdgeToolsGenerationLoop(
        tokenizer: tokenizer,
        extraStopTokenIds: extraStopTokenIds
      )
      self.tokenizer = tokenizer
    }

    public func context() -> MLXContext<Profile> {
      self.context(MLXContextParameters(), tools: [])
    }

    public func context(tools: [any EdgeTool]) -> MLXContext<Profile> {
      self.context(MLXContextParameters(), tools: tools)
    }

    public func context(
      _ parameters: MLXContextParameters,
      tools: [any EdgeTool]
    ) -> MLXContext<Profile> {
      MLXContext(
        parameters: EdgeToolsTranscriptContextParameters(
          transcript: parameters.transcript,
          reasoningEffort: parameters.reasoningEffort
        ),
        tools: tools,
        model: MLXModelState<Profile>(policy: MLXCachePolicy(parameters: parameters)),
        engineIdentity: self.identity
      )
    }

    public func tokenize(
      prompt: EdgeToolsTranscript.Prompt,
      context: MLXContext<Profile>
    ) async throws -> [EdgeToolsToken] {
      try self.validate(context)
      let transcript = context.transcript(appending: prompt)
      let tokenIds = try await self.inputProcessor.tokenIds(
        prompt: transcript,
        tools: context.tools.map { $0.definition }
      )
      return self.tokenizer.tokens(forIds: tokenIds).compactMap { $0 }
    }

    public func generate(
      prompt: EdgeToolsTranscript.Prompt,
      parameters: sending Profile.GenerateParameters,
      context: MLXContext<Profile>,
      channel: sending EdgeToolsGenerationChannel
    ) throws -> AnyGenerationTask {
      try self.validate(context)
      return self.generationTask(
        tools: context.tools.map { $0.definition },
        parameters: parameters,
        context: context,
        channel: channel,
        snapshot: { try context.begin(appending: prompt) }
      )
    }

    public func prefill(
      promptPrefix: EdgeToolsTranscript.Prompt,
      context: MLXContext<Profile>
    ) async throws -> EdgeToolsEnginePrefill {
      try self.validate(context)
      return try await self.prefill(
        snapshot: context.begin(appending: promptPrefix),
        tools: context.tools.map { $0.definition },
        context: context
      )
    }

    public func prefill(context: MLXContext<Profile>) async throws -> EdgeToolsEnginePrefill {
      try self.validate(context)
      let definitions = context.tools.map { $0.definition }
      return try await self.prefill(snapshot: context.begin(), tools: definitions, context: context)
    }

    public func generate(
      parameters: sending Profile.GenerateParameters,
      context: MLXContext<Profile>,
      channel: sending EdgeToolsGenerationChannel
    ) throws -> AnyGenerationTask {
      try self.validate(context)
      return self.generationTask(
        tools: context.tools.map { $0.definition },
        parameters: parameters,
        context: context,
        channel: channel,
        snapshot: { try context.begin() }
      )
    }

    private func generationTask(
      tools: [EdgeToolDefinition],
      parameters: sending Profile.GenerateParameters,
      context: MLXContext<Profile>,
      channel: sending EdgeToolsGenerationChannel,
      snapshot: @escaping @Sendable () throws -> MLXContext<Profile>.Snapshot
    ) -> AnyGenerationTask {
      return AnyGenerationTask { stopper in
        let snapshot = try snapshot()
        var model = snapshot.model
        do {
          let result = try await self.runtime.generate(
            prompt: snapshot.transcript,
            tools: tools,
            parameters: parameters,
            configuredSampling: self.configuredSampling,
            generationLoop: self.generationLoop,
            grammarEngine: self.grammarEngine,
            stopper: stopper,
            channel: channel,
            checkpoint: model.checkpoint,
            policy: model.policy
          )
          model.checkpoint = result.checkpoint
          context.finish(
            generation: result.generation,
            revision: snapshot.revision,
            model: model
          )
          return result.generation
        } catch {
          context.finish(generation: nil, revision: snapshot.revision, model: model)
          throw error
        }
      }
    }

    private func prefill(
      snapshot: MLXContext<Profile>.Snapshot,
      tools: [EdgeToolDefinition],
      context: MLXContext<Profile>
    ) async throws -> EdgeToolsEnginePrefill {
      var model = snapshot.model
      defer { context.finish(generation: nil, revision: snapshot.revision, model: model) }
      let result = try await self.runtime.prefill(
        prompt: snapshot.transcript,
        tools: tools,
        checkpoint: model.checkpoint,
        policy: model.policy
      )
      model.checkpoint = result.checkpoint
      return result.prefill
    }

    private func validate(_ context: MLXContext<Profile>) throws {
      guard context.engineIdentity === self.identity else {
        throw EdgeToolsError.incompatibleContext
      }
    }

  }

  #if XGrammar
    extension MLXEngine where Profile.GrammarEngine == XGrammarEngine {
      public func clearCaches() {
        self.grammarEngine.clearCaches()
      }
    }
  #endif

  // MARK: - EdgeToolsSession + MLX

  #if XGrammar
    extension EdgeToolsSession {
      public func clearCaches<Profile>()
      where
        Engine == MLXEngine<Profile>,
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
    tokenIds.formUnion(Profile.extraStopTokens.compactMap { tokenizer.token(forText: $0)?.id })
    if let eosTokenId = tokenizer.eos?.id { tokenIds.remove(eosTokenId) }
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
        .decode(BaseProcessorConfiguration.self, from: data)
      let processorType =
        switch modelType {
        case "mistral3": "Mistral3Processor"
        case "gemma4_unified": "Gemma4UnifiedProcessor"
        default: configuration.processorClass
        }
      return try await VLMProcessorTypeRegistry.shared.createModel(
        configuration: data,
        processorType: processorType,
        tokenizer: tokenizer.mlxTokenizer
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

#endif
