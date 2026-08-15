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

  // MARK: - MLXGenerationState

  public struct MLXGenerationState<Profile: MLXModelProfile>
  where Profile.Prompt == EdgeToolsTranscript {
    public var model: MLXModelState<Profile>
    public var transcript: EdgeToolsTranscript
    public let revision: Int

    public init(
      model: MLXModelState<Profile>,
      transcript: EdgeToolsTranscript,
      revision: Int
    ) {
      self.model = model
      self.transcript = transcript
      self.revision = revision
    }
  }

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
#endif
