#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && canImport(MLX)
  import EdgeToolsCore
  import MLX
  import MLXLMCommon

  struct MLXGenerationTransaction {
    var transcript: EdgeToolsTranscript
    let reasoningEffort: EdgeToolsReasoningEffort
    var generation: MLXGeneration?
  }

  // MARK: - MLXRuntime

  actor MLXRuntime<Profile: MLXModelProfile> where Profile.Prompt == EdgeToolsTranscript {
    private let languageModel: any LanguageModel
    private let inputProcessor: MLXInputProcessor<Profile>
    private var checkpoints = [UInt64: MLXPrefixCache]()
    private var nextCheckpointID: UInt64 = 0

    init(
      languageModel: sending any LanguageModel,
      inputProcessor: MLXInputProcessor<Profile>
    ) {
      self.languageModel = languageModel
      self.inputProcessor = inputProcessor
    }

    func generate(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      parameters: MLXGenerateParameters,
      configuredSampling: EdgeToolsFusedSamplingParameters?,
      generationLoop: EdgeToolsGenerationLoop,
      grammarEngine: Profile.GrammarEngine,
      stopper: AnyGenerationTask.Stopper,
      channel: EdgeToolsGenerationChannel,
      checkpoint: MLXPrefixCacheHandle?,
      policy: MLXCachePolicy
    ) async throws -> MLXGenerationResult {
      try Task.checkCancellation()
      guard !stopper.isStopped else {
        return MLXGenerationResult(generation: .empty, checkpoint: checkpoint)
      }
      var transaction = MLXGenerationTransaction(
        transcript: prompt,
        reasoningEffort: reasoningEffort
      )
      var parser = Profile.GenerationParser()
      Profile.prepare(
        prompt: &transaction.transcript,
        reasoningEffort: reasoningEffort,
        tools: tools,
        parser: &parser
      )
      let defaultSampling =
        Profile.defaultSampling(
          prompt: transaction.transcript,
          reasoningEffort: reasoningEffort
        )
        ?? configuredSampling
        ?? EdgeToolsFusedSamplingParameters()
      let prepared = try await self.prepareGeneration(
        prompt: transaction.transcript,
        reasoningEffort: reasoningEffort,
        tools: tools,
        sampler: parameters.sampler?()
          ?? MLXFusedSampler(parameters: parameters.sampling.applying(to: defaultSampling)),
        processor: parameters.processor?(),
        confidenceOptions: parameters.confidence,
        synchronizeMemorySnapshots: parameters.synchronizeStreamForMemorySnapshots,
        checkpoint: checkpoint,
        policy: policy
      )
      transaction.generation = prepared.generation
      var result = try generationLoop.runPrepared(
        state: &transaction,
        parser: &parser,
        preparation: EdgeToolsGenerationLoop.Preparation(metrics: prepared.metrics),
        stopper: stopper,
        channel: channel,
        grammarEngine: grammarEngine,
        maximumTokenCount: parameters.maxTokens,
        grammar: {
          try Profile.grammar(
            prompt: $0.transcript,
            reasoningEffort: $0.reasoningEffort,
            tools: tools,
            constraint: parameters.constraint,
            grammarEngine: grammarEngine
          )
        },
        decode: { bitmask, transaction in
          guard var generation = transaction.generation else {
            throw EdgeToolsError.modelNotPrepared
          }
          let tokenId = try self.decode(&generation, bitmask: bitmask, policy: policy)
          transaction.generation = generation
          return tokenId
        }
      )
      guard var generation = transaction.generation else {
        throw EdgeToolsError.modelNotPrepared
      }
      let metrics = self.metrics(for: generation)
      let checkpoint = self.commit(
        &generation,
        stopTokenIds: generationLoop.stopTokenIds,
        policy: policy
      )
      result.metrics.merge(metrics) { _, finalValue in finalValue }
      return MLXGenerationResult(generation: result, checkpoint: checkpoint)
    }

    func prefill(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      checkpoint: MLXPrefixCacheHandle?,
      policy: MLXCachePolicy
    ) async throws -> MLXPrefillResult {
      let preparedInput = try await self.input(
        prompt: prompt,
        reasoningEffort: reasoningEffort,
        tools: tools,
        kind: .prefill,
        checkpoint: checkpoint
      )
      let clock = ContinuousClock()
      let tokenIds = preparedInput.input.text.tokens.asArray(EdgeToolsToken.ID.self)
      let start = clock.now
      let prepared = try self.prefixState(
        input: preparedInput.input,
        context: preparedInput.context,
        tokenIds: tokenIds,
        checkpoint: checkpoint,
        policy: policy
      )
      eval(prepared.state.output.logits)
      eval(prepared.state.cache)
      let checkpoint = self.store(MLXPrefixCache(state: prepared.state, inputKind: .prefill))
      var metrics = EdgeToolsMetrics()
      metrics.mlxEnginePostPrefillMemorySnapshot = self.memorySnapshot(synchronize: true)
      metrics.prefillTokens = prepared.tokenCount
      metrics.prefillDuration = start.duration(to: clock.now)
      let prefill = EdgeToolsEnginePrefill(metrics: metrics)
      return MLXPrefillResult(prefill: prefill, checkpoint: checkpoint)
    }

    private func memorySnapshot(synchronize: Bool) -> Memory.Snapshot {
      if synchronize {
        Stream.defaultStream(.defaultDevice()).synchronize()
      }
      return Memory.snapshot()
    }

    private func prepareGeneration(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      sampler: any LogitSampler,
      processor: (any LogitProcessor)?,
      confidenceOptions: EdgeToolsConfidenceOptions,
      synchronizeMemorySnapshots: Bool,
      checkpoint: MLXPrefixCacheHandle?,
      policy: MLXCachePolicy
    ) async throws -> (generation: MLXGeneration, metrics: EdgeToolsMetrics) {
      let preparedInput = try await self.input(
        prompt: prompt,
        reasoningEffort: reasoningEffort,
        tools: tools,
        kind: .generation,
        checkpoint: checkpoint
      )
      let clock = ContinuousClock()
      let generationStartSnapshot = self.memorySnapshot(synchronize: synchronizeMemorySnapshots)
      let tokenIds = preparedInput.input.text.tokens.asArray(EdgeToolsToken.ID.self)
      let start = clock.now
      var processor = processor
      processor?.prompt(preparedInput.input.text.tokens)
      (sampler as? MLXFusedSampler)?.history.seed(tokenIds)
      let prepared = try self.prefixState(
        input: preparedInput.input,
        context: preparedInput.context,
        tokenIds: tokenIds,
        checkpoint: checkpoint,
        policy: policy
      )
      var metrics = EdgeToolsMetrics()
      metrics.prefillTokens = prepared.tokenCount
      metrics.prefillDuration = start.duration(to: clock.now)
      let generation = MLXGeneration(
        prefix: prepared.state,
        decoder: DecoderState(sampler: sampler, confidenceOptions: confidenceOptions),
        processor: processor,
        synchronizeStreamForMemorySnapshots: synchronizeMemorySnapshots,
        generationStartSnapshot: generationStartSnapshot,
        postPrefillSnapshot: self.memorySnapshot(synchronize: synchronizeMemorySnapshots)
      )
      return (generation, metrics)
    }

    private func decode(
      _ generation: inout MLXGeneration,
      bitmask: GrammarBitmask?,
      policy: MLXCachePolicy
    ) throws -> EdgeToolsToken.ID {
      if let pendingTokenId = generation.decoder.pendingTokenId {
        self.extend(&generation.prefix, with: pendingTokenId, policy: policy)
      }
      var stepLogits = generation.prefix.output.logits[0..., -1, 0...]
      stepLogits = generation.processor?.process(logits: stepLogits) ?? stepLogits
      let maskedLogits =
        bitmask.map { applyBitmaskMLX(logits: stepLogits, mask: $0) }
        ?? stepLogits
      let token = generation.decoder.sampler.sample(logits: maskedLogits)
      if generation.decoder.tracksTokenConfidence {
        let confidenceValues = top(maskedLogits.flattened(), k: 2)
        eval(confidenceValues, token)

        let confidence = tokenConfidence(unorderedPair: confidenceValues.asArray(Float.self))
        generation.decoder.add(confidence: confidence)
      }
      let tokenId = token.item(EdgeToolsToken.ID.self)
      generation.processor?.didSample(token: token)
      generation.decoder.pendingTokenId = tokenId
      return tokenId
    }

    private func metrics(for generation: MLXGeneration) -> EdgeToolsMetrics {
      var metrics = generation.decoder.metrics
      metrics.mlxEngineGenerationStartMemorySnapshot = generation.generationStartSnapshot
      metrics.mlxEnginePostPrefillMemorySnapshot = generation.postPrefillSnapshot
      metrics.mlxEnginePostDecodeMemorySnapshot = self.memorySnapshot(
        synchronize: generation.synchronizeStreamForMemorySnapshots
      )
      return metrics
    }

    private func commit(
      _ generation: inout MLXGeneration,
      stopTokenIds: Set<EdgeToolsToken.ID>,
      policy: MLXCachePolicy
    ) -> MLXPrefixCacheHandle {
      if let pendingTokenId = generation.decoder.pendingTokenId,
        !stopTokenIds.contains(pendingTokenId)
      {
        self.extend(&generation.prefix, with: pendingTokenId, policy: policy)
      }
      eval(generation.prefix.output.logits)
      eval(generation.prefix.cache)
      return self.store(MLXPrefixCache(state: generation.prefix, inputKind: .generation))
    }

    private func input(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      kind: EdgeToolsLLMInputKind,
      checkpoint: MLXPrefixCacheHandle?
    ) async throws -> (input: LMInput, context: EdgeToolsLLMPrefillContext) {
      let context = EdgeToolsLLMPrefillContext(
        prompt: prompt,
        reasoningEffort: reasoningEffort,
        tools: tools
      )
      if let checkpoint,
        let input = self.checkpoints[checkpoint.id]?.preparedInput(for: context, kind: kind)
      {
        return (input, context)
      }
      let input = try await self.inputProcessor.input(
        prompt: prompt,
        reasoningEffort: reasoningEffort,
        tools: tools,
        kind: kind
      )
      return (input, context)
    }

    private func extend(
      _ state: inout MLXPrefixState,
      with tokenId: EdgeToolsToken.ID,
      policy: MLXCachePolicy
    ) {
      state.output = self.languageModel(
        LMInput.Text(tokens: MLXArray([tokenId]))[text: .newAxis],
        cache: state.cache,
        state: state.output.state
      )
      state.tokenIds.append(tokenId)
      quantizeCache(&state.cache, quantization: policy.quantization)
    }

    private func prefixState(
      input: LMInput,
      context: EdgeToolsLLMPrefillContext,
      tokenIds: [EdgeToolsToken.ID],
      checkpoint: MLXPrefixCacheHandle?,
      policy: MLXCachePolicy
    ) throws -> (state: MLXPrefixState, tokenCount: Int) {
      func state(output: LMOutput, cache: [any KVCache]) -> MLXPrefixState {
        MLXPrefixState(
          input: input,
          tokenIds: tokenIds,
          cache: cache,
          output: output,
          context: context
        )
      }
      guard
        let checkpoint,
        let snapshot = self.checkpoints[checkpoint.id]?.state(
          continuingWith: tokenIds,
          input: input,
          context: context
        )
      else {
        let parameters = policy.maxKVSize.map { MLXLMCommon.GenerateParameters(maxKVSize: $0) }
        var cache = self.languageModel.newCache(parameters: parameters)
        let output = try self.prepareModelOutput(
          input: input,
          cache: cache,
          chunkSize: policy.prefillChunkSize
        )
        quantizeCache(&cache, quantization: policy.quantization)
        return (state(output: output, cache: cache), tokenIds.count)
      }
      let suffixCount = tokenIds.count - snapshot.tokenIds.count
      guard suffixCount > 0 else {
        return (state(output: snapshot.output, cache: snapshot.cache), suffixCount)
      }
      var cache = snapshot.cache
      let output = self.evaluateText(
        mlxTextSuffix(input.text, from: snapshot.tokenIds.count),
        cache: cache,
        outputState: snapshot.output.state,
        chunkSize: policy.prefillChunkSize
      )
      quantizeCache(&cache, quantization: policy.quantization)
      return (state(output: output, cache: cache), suffixCount)
    }

    private func store(_ prefixCache: MLXPrefixCache) -> MLXPrefixCacheHandle {
      self.nextCheckpointID += 1
      let id = self.nextCheckpointID
      self.checkpoints[id] = prefixCache
      return MLXPrefixCacheHandle(id: id) { [weak self] id in
        Task { await self?.removeCheckpoint(id: id) }
      }
    }

    private func removeCheckpoint(id: UInt64) {
      self.checkpoints.removeValue(forKey: id)
    }

    private func prepareModelOutput(
      input: LMInput,
      cache: [any KVCache],
      chunkSize: Int
    ) throws -> LMOutput {
      switch try self.languageModel.prepare(input, cache: cache, windowSize: chunkSize) {
      case .logits(let output):
        return output
      case .tokens(let tokens):
        guard tokens.tokens.size > 0 else {
          throw MLXEngineError(code: .emptyInput, message: "Model received empty input.")
        }
        return self.evaluateText(tokens, cache: cache, outputState: nil, chunkSize: chunkSize)
      }
    }

    private func evaluateText(
      _ text: LMInput.Text,
      cache: [any KVCache],
      outputState: LMOutput.State?,
      chunkSize: Int
    ) -> LMOutput {
      var remaining = text
      var state = outputState
      return withPreparedCache(cache, lengths: text.sequenceLengths) {
        while remaining.tokens.dim(-1) > chunkSize {
          let output = self.languageModel(
            mlxTextPrefix(remaining, count: chunkSize),
            cache: cache.isEmpty ? nil : cache,
            state: state
          )
          state = output.state
          asyncEval(cache)
          remaining = mlxTextSuffix(remaining, from: chunkSize)
        }
        return self.languageModel(
          mlxTextPrefix(remaining, count: remaining.tokens.dim(-1)),
          cache: cache.isEmpty ? nil : cache,
          state: state
        )
      }
    }
  }

  // MARK: - Helpers

  private func quantizeCache(_ cache: inout [any KVCache], quantization: MLXKVCacheQuantization?) {
    guard let quantization else { return }
    maybeQuantizeKVCache(
      cache: &cache,
      kvBits: quantization.bits,
      kvGroupSize: quantization.groupSize,
      quantizedKVStart: quantization.startTokenCount
    )
  }
#endif
