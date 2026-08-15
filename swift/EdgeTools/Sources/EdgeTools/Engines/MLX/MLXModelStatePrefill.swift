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

  // MARK: - MLXModelState Prefill

  extension MLXModelState {
    public mutating func commitGeneration(
      stopTokenIds: Set<EdgeToolsToken.ID>
    ) {
      guard var generation = self.generation else { return }
      if let pendingTokenId = generation.pendingTokenId,
        !stopTokenIds.contains(pendingTokenId)
      {
        let output = self.languageModel(
          LMInput.Text(tokens: MLXArray([pendingTokenId]))[text: .newAxis],
          cache: generation.cache,
          state: generation.outputState
        )
        generation.outputState = output.state
        generation.logits = output.logits
        generation.cachedTokenIds.append(pendingTokenId)
      }
      eval(generation.logits)
      eval(generation.cache)
      self.prefillCacheState.cachedPrefill = CachedPrefill(
        input: generation.input,
        tokenIds: generation.cachedTokenIds,
        cache: generation.cache,
        output: LMOutput(logits: generation.logits, state: generation.outputState),
        context: generation.inputContext,
        inputKind: .generation
      )
      self.generation = nil
    }

    public mutating func resetGeneration() async {
      self.generation = nil
    }

    public func contextState() -> sending Self {
      // Model weights are immutable after engine initialization and intentionally shared across
      // contexts. All mutable generation and prefill state is created afresh below.
      nonisolated(unsafe) let languageModel = self.languageModel
      return Self(
        languageModel: languageModel,
        processor: self.processor,
        vocabularySize: self.vocabularySizeValue,
        extraStopTokenIds: self.configuredExtraStopTokenIds,
        defaultSampling: self.configuredSampling
      )
    }

    fileprivate func forkedContextState(copyingCache: Bool) -> sending Self {
      // Model weights are immutable after engine initialization and intentionally shared across
      // contexts. The cached prefill is immutable, or is copied here for an active-context fork.
      nonisolated(unsafe) let languageModel = self.languageModel
      var state = Self(
        languageModel: languageModel,
        processor: self.processor,
        vocabularySize: self.vocabularySizeValue,
        extraStopTokenIds: self.configuredExtraStopTokenIds,
        defaultSampling: self.configuredSampling
      )
      state.prefillCacheState = self.prefillCacheState
      state.prefillCacheState.fork(copyingCache: copyingCache)
      // The returned state shares only immutable model weights and, for an idle context, an
      // immutable cached-prefill checkpoint whose cache copies are serialized internally. Swift's
      // isolation checker cannot express that ownership boundary.
      nonisolated(unsafe) let transferredState = state
      return transferredState
    }

    public nonisolated(nonsending) mutating func prefill(
      prompt: Profile.Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer
    ) async throws -> EdgeToolsEnginePrefill {
      let input = try await self.input(
        prompt: prompt,
        tools: tools,
        tokenizer: tokenizer,
        kind: .prefill
      )
      return try await self.prefill(input: input)
    }

    private nonisolated(nonsending) mutating func prefill(
      input: LMInput
    ) async throws -> EdgeToolsEnginePrefill {
      let clock = ContinuousClock()
      let tokenIds = input.text.tokens.asArray(EdgeToolsToken.ID.self)
      let start = clock.now
      let prepared = try self.preparedOutput(input: input, tokenIds: tokenIds)
      eval(prepared.output.logits)
      eval(prepared.cache)
      self.prefillCacheState.cachedPrefill = CachedPrefill(
        input: input,
        tokenIds: tokenIds,
        cache: prepared.cache,
        output: prepared.output,
        context: self.prefillCacheState.inputContext,
        inputKind: .prefill
      )
      let snapshot = Self.memorySnapshot(synchronize: true)
      var metadata = EdgeToolsMetadata()
      metadata.mlxEnginePostPrefillMemorySnapshot = snapshot
      return EdgeToolsEnginePrefill(
        metrics: EdgeToolsPrefillMetrics(
          tokens: prepared.tokenCount,
          duration: start.duration(to: clock.now)
        ),
        metadata: metadata
      )
    }

    private nonisolated(nonsending) mutating func input(
      prompt: Profile.Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      kind: MLXInputKind
    ) async throws -> LMInput {
      if let prompt = prompt as? EdgeToolsTranscript {
        let context = EdgeToolsLLMPrefillContext(prompt: prompt, tools: tools)
        if let input = self.prefillCacheState.input(for: context, kind: kind) {
          return input
        }
      } else {
        self.prefillCacheState.clearInputContext()
      }
      switch kind {
      case .generation:
        return try await Profile.input(
          prompt: prompt,
          tools: tools,
          tokenizer: tokenizer,
          processor: self.processor
        )
      case .prefill:
        return try await Profile.prefillInput(
          prompt: prompt,
          tools: tools,
          tokenizer: tokenizer,
          processor: self.processor
        )
      }
    }

    private func preparedOutput(
      input: LMInput,
      tokenIds: [EdgeToolsToken.ID]
    ) throws -> (output: LMOutput, cache: [any KVCache], tokenCount: Int) {
      guard
        let cachedPrefill = self.prefillCacheState.cachedPrefill?.mutableSnapshot(
          tokenIds: tokenIds,
          input: input,
          inputContext: self.prefillCacheState.inputContext
        )
      else {
        let cache = self.languageModel.newCache(parameters: nil)
        return (try self.prepareModelOutput(input: input, cache: cache), cache, tokenIds.count)
      }
      let suffixCount = tokenIds.count - cachedPrefill.tokenIds.count
      let cache = cachedPrefill.cache
      let output =
        if suffixCount == 0 {
          cachedPrefill.output
        } else {
          self.languageModel(
            mlxTextSuffix(input.text, from: cachedPrefill.tokenIds.count),
            cache: cache,
            state: cachedPrefill.output.state
          )
        }
      return (output, cache, suffixCount)
    }

    private func prepareModelOutput(input: LMInput, cache: [any KVCache]) throws -> LMOutput {
      switch try self.languageModel.prepare(input, cache: cache, windowSize: nil) {
      case .logits(let output):
        return output
      case .tokens(let tokens):
        guard tokens.tokens.size > 0 else {
          throw MLXEngineError(code: .emptyInput, message: "Model received empty input.")
        }
        return self.languageModel(
          tokens[text: .newAxis],
          cache: cache.isEmpty ? nil : cache,
          state: nil
        )
      }
    }

    private static func memorySnapshot(synchronize: Bool) -> Memory.Snapshot {
      if synchronize {
        Stream.defaultStream(.defaultDevice()).synchronize()
      }
      return Memory.snapshot()
    }
  }
#endif
