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

  // MARK: - MLXEngine

  public final class MLXEngine<Profile: MLXModelProfile>:
    EdgeToolsEngine, EdgeToolsPrefillableEngine
  where Profile.Prompt == EdgeToolsTranscript {
    public typealias Context = MLXContext<Profile>
    public typealias ContextParameters = MLXContextParameters
    public typealias Prompt = EdgeToolsTranscript.UserMessage
    public typealias GenerateParameters = Profile.GenerateParameters
    public typealias ModelGenerationState = MLXGenerationState<Profile>
    public typealias GenerationParser = Profile.GenerationParser
    public typealias GrammarEngine = Profile.GrammarEngine

    private let prototype: Lock<MLXModelState<Profile>>
    private let identity = MLXEngineIdentity()
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
      let prototype = MLXModelState<Profile>(
        languageModel: languageModel,
        processor: processor,
        vocabularySize: vocabularySize,
        extraStopTokenIds: extraStopTokenIds,
        defaultSampling: defaultSampling
      )
      self.grammarEngine = try prototype.grammarEngine(tokenizer: tokenizer)
      self.prototype = Lock(prototype)
      self.generationLoop = EdgeToolsGenerationLoop(
        tokenizer: tokenizer,
        extraStopTokenIds: extraStopTokenIds
      )
      self.tokenizer = tokenizer
    }

    public func context() -> MLXContext<Profile> {
      self.context(MLXContextParameters())
    }

    public func context(_ parameters: MLXContextParameters) -> MLXContext<Profile> {
      MLXContext(
        parameters: parameters,
        model: self.makeModelState(),
        engineIdentity: self.identity
      )
    }

    public func tokenize(
      prompt: EdgeToolsTranscript.UserMessage,
      tools: [EdgeToolDefinition],
      context: MLXContext<Profile>
    ) async throws -> [EdgeToolsToken] {
      try self.validate(context)
      let transcript = context.transcript(appending: prompt)
      var model = self.makeModelState()
      let tokenIds = try await model.tokenIds(
        prompt: transcript,
        tools: tools,
        tokenizer: self.tokenizer
      )
      let tokens = self.tokenizer.convertIdsToTokens(tokenIds)
      return zip(tokenIds, tokens)
        .compactMap { tokenId, token in
          token.map { EdgeToolsToken(id: tokenId, stringValue: $0) }
        }
    }

    public func generationState(
      prompt: EdgeToolsTranscript.UserMessage,
      context: MLXContext<Profile>
    ) async throws -> ModelGenerationState {
      try self.validate(context)
      return self.generationState(from: try context.begin(appending: prompt))
    }

    public func generate(
      prompt: EdgeToolsTranscript.UserMessage,
      tools: [EdgeToolDefinition] = [],
      parameters: sending Profile.GenerateParameters,
      context: MLXContext<Profile>,
      channel: sending EdgeToolsGenerationChannel
    ) throws -> AnyGenerationTask {
      self.generationTask(
        prompt: prompt,
        tools: tools,
        parameters: parameters,
        context: context,
        channel: channel,
        makeState: {
          try await self.generationState(prompt: prompt, context: context)
        }
      )
    }

    public func prepare(
      prompt: inout EdgeToolsTranscript.UserMessage,
      tools: [EdgeToolDefinition],
      parameters: Profile.GenerateParameters,
      parser: inout Profile.GenerationParser,
      state: inout ModelGenerationState
    ) async throws -> EdgeToolsGenerationLoop.Preparation {
      try await state.model.prepare(
        prompt: &state.transcript,
        tools: tools,
        tokenizer: self.tokenizer,
        parameters: parameters,
        parser: &parser
      )
    }

    public func grammar(
      prompt: EdgeToolsTranscript.UserMessage,
      tools: [EdgeToolDefinition],
      parameters: Profile.GenerateParameters,
      state: ModelGenerationState
    ) throws -> Profile.GrammarEngine.Grammar {
      try state.model.grammar(
        prompt: state.transcript,
        tools: tools,
        parameters: parameters,
        grammarEngine: self.grammarEngine
      )
    }

    public func decode(
      bitmask: GrammarBitmask,
      parameters: Profile.GenerateParameters,
      state: inout ModelGenerationState
    ) async throws -> EdgeToolsToken.ID {
      try await state.model.decode(bitmask: bitmask, parameters: parameters)
    }

    public func finalize(
      state: consuming ModelGenerationState,
      result: consuming Result<EdgeToolsEngineGeneration, any Error>,
      context: MLXContext<Profile>
    ) async -> Result<EdgeToolsEngineGeneration, any Error> {
      var state = state
      let metadata = state.model.finish()

      let generation: EdgeToolsEngineGeneration?
      let finalResult: Result<EdgeToolsEngineGeneration, any Error>
      switch result {
      case .success(var value):
        state.model.commitGeneration(stopTokenIds: self.generationLoop.stopTokenIds)
        value.metadata.merge(metadata) { _, finalValue in finalValue }
        generation = value
        finalResult = .success(value)
      case .failure(let error):
        await state.model.resetGeneration()
        generation = nil
        finalResult = .failure(error)
      }
      // The state is returned exactly once and is not accessed after this point. Region isolation
      // does not currently infer that exclusivity through the synchronous context method.
      nonisolated(unsafe) let restoredModel = state.model
      context.finish(
        generation: generation,
        revision: state.revision,
        model: restoredModel
      )
      return finalResult
    }

    public func prefill(
      promptPrefix: EdgeToolsTranscript.UserMessage,
      tools: [EdgeToolDefinition],
      context: MLXContext<Profile>
    ) async throws -> EdgeToolsEnginePrefill {
      try self.validate(context)
      return try await self.prefill(
        snapshot: context.begin(appending: promptPrefix),
        tools: tools,
        context: context
      )
    }

    public func prefill(
      tools: [EdgeToolDefinition] = [],
      context: MLXContext<Profile>
    ) async throws -> EdgeToolsEnginePrefill {
      try self.validate(context)
      return try await self.prefill(
        snapshot: context.begin(),
        tools: tools,
        context: context
      )
    }

    public func generate(
      tools: [EdgeToolDefinition] = [],
      parameters: sending Profile.GenerateParameters,
      context: MLXContext<Profile>,
      channel: sending EdgeToolsGenerationChannel
    ) throws -> AnyGenerationTask {
      try self.validate(context)
      return self.generationTask(
        prompt: .user(""),
        tools: tools,
        parameters: parameters,
        context: context,
        channel: channel,
        makeState: {
          self.generationState(from: try context.begin())
        }
      )
    }

    private func generationTask(
      prompt: EdgeToolsTranscript.UserMessage,
      tools: [EdgeToolDefinition],
      parameters: sending Profile.GenerateParameters,
      context: MLXContext<Profile>,
      channel: sending EdgeToolsGenerationChannel,
      makeState: @escaping @Sendable () async throws -> ModelGenerationState
    ) -> AnyGenerationTask {
      AnyGenerationTask { stopper in
        var state = try await makeState()
        var preparedPrompt = prompt
        let result: Result<EdgeToolsEngineGeneration, any Error>
        do {
          result = .success(
            try await self.generationLoop.run(
              state: &state,
              stopper: stopper,
              channel: channel,
              grammarEngine: self.grammarEngine,
              maximumTokenCount: parameters.maxTokens,
              grammar: { state in
                try self.grammar(
                  prompt: preparedPrompt,
                  tools: tools,
                  parameters: parameters,
                  state: state
                )
              },
              prepare: { parser, state in
                return try await self.prepare(
                  prompt: &preparedPrompt,
                  tools: tools,
                  parameters: parameters,
                  parser: &parser,
                  state: &state
                )
              },
              decode: { bitmask, state in
                try await self.decode(
                  bitmask: bitmask,
                  parameters: parameters,
                  state: &state
                )
              }
            )
          )
        } catch {
          result = .failure(error)
        }
        return try await self.finalize(state: state, result: result, context: context).get()
      }
    }

    private func prefill(
      snapshot: MLXContext<Profile>.GenerationSnapshot,
      tools: [EdgeToolDefinition],
      context: MLXContext<Profile>
    ) async throws -> EdgeToolsEnginePrefill {
      var model = snapshot.model
      do {
        let prefill = try await model.prefill(
          prompt: snapshot.transcript,
          tools: tools,
          tokenizer: self.tokenizer
        )
        // The model is no longer used after it is returned to the context. Region isolation does
        // not currently infer that exclusivity through this synchronous transfer.
        nonisolated(unsafe) let restoredModel = model
        context.finish(generation: nil, revision: snapshot.revision, model: restoredModel)
        return prefill
      } catch {
        await model.resetGeneration()
        nonisolated(unsafe) let restoredModel = model
        context.finish(generation: nil, revision: snapshot.revision, model: restoredModel)
        throw error
      }
    }

    private func makeModelState() -> sending MLXModelState<Profile> {
      self.prototype.withBorrowedLock { $0.contextState() }
    }

    private func validate(_ context: MLXContext<Profile>) throws {
      guard context.engineIdentity === self.identity else {
        throw EdgeToolsError.incompatibleContext
      }
    }

    private func generationState(
      from snapshot: MLXContext<Profile>.GenerationSnapshot
    ) -> ModelGenerationState {
      ModelGenerationState(
        model: snapshot.model,
        transcript: snapshot.transcript,
        revision: snapshot.revision
      )
    }
  }

  #if XGrammar
    extension MLXEngine where Profile.GrammarEngine == XGrammarEngine {
      public func clearCaches() {
        self.grammarEngine.clearCaches()
      }
    }
  #endif
#endif
