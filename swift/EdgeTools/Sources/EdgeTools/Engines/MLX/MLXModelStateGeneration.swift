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

  // MARK: - MLXModelState Generation

  extension MLXModelState {
    public var vocabularySize: Int { self.vocabularySizeValue }
    public var extraStopTokenIds: Set<EdgeToolsToken.ID> {
      self.configuredExtraStopTokenIds
    }

    public func grammarEngine(
      tokenizer: any EdgeToolsTokenizer
    ) throws -> Profile.GrammarEngine {
      var stopTokenIds = self.extraStopTokenIds
      if let eosTokenId = tokenizer.eosTokenId { stopTokenIds.insert(eosTokenId) }
      return try Profile.grammarEngine(
        tokenizer: tokenizer,
        vocabularySize: self.vocabularySize,
        stopTokenIds: stopTokenIds
      )
    }

    public func grammar(
      prompt: Profile.Prompt,
      tools: [EdgeToolDefinition],
      parameters: Profile.GenerateParameters,
      grammarEngine: borrowing Profile.GrammarEngine
    ) throws -> Profile.GrammarEngine.Grammar {
      try Profile.grammar(
        prompt: prompt,
        tools: tools,
        parameters: parameters,
        grammarEngine: grammarEngine
      )
    }

    public nonisolated(nonsending) mutating func prepare(
      prompt: inout Profile.Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      parameters: Profile.GenerateParameters,
      parser: inout Profile.GenerationParser
    ) async throws -> EdgeToolsGenerationLoop.Preparation {
      Profile.prepare(prompt: &prompt, tools: tools, parser: &parser)
      let input = try await self.input(
        prompt: prompt,
        tools: tools,
        tokenizer: tokenizer,
        kind: .generation
      )
      let defaultSampling =
        Profile.defaultSampling(prompt: prompt, parameters: parameters)
        ?? self.configuredSampling
        ?? EdgeToolsFusedSamplingParameters()
      let sampler =
        parameters.sampler
        ?? MLXFusedSampler(parameters: parameters.sampling.applying(to: defaultSampling))
      return try await self.prepare(input: input, sampler: sampler, parameters: parameters)
    }

    public nonisolated(nonsending) mutating func tokenIds(
      prompt: Profile.Prompt,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer
    ) async throws -> [EdgeToolsToken.ID] {
      let input = try await self.input(
        prompt: prompt,
        tools: tools,
        tokenizer: tokenizer,
        kind: .generation
      )
      return input.text.tokens.asArray(EdgeToolsToken.ID.self)
    }

    private nonisolated(nonsending) mutating func prepare(
      input: LMInput,
      sampler: any LogitSampler,
      parameters: Profile.GenerateParameters
    ) async throws -> EdgeToolsGenerationLoop.Preparation {
      let clock = ContinuousClock()
      let generationStartSnapshot = Self.memorySnapshot(
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      let tokenIds = input.text.tokens.asArray(EdgeToolsToken.ID.self)
      let start = clock.now
      var processor = parameters.processor
      processor?.prompt(input.text.tokens)
      let prepared = try self.preparedOutput(input: input, tokenIds: tokenIds)
      let metrics = EdgeToolsPrefillMetrics(
        tokens: prepared.tokenCount,
        duration: start.duration(to: clock.now)
      )
      let postPrefillSnapshot = Self.memorySnapshot(
        synchronize: parameters.synchronizeStreamForMemorySnapshots
      )
      self.generation = Generation(
        input: input,
        cachedTokenIds: tokenIds,
        cache: prepared.cache,
        outputState: prepared.output.state,
        logits: prepared.output.logits,
        pendingTokenId: nil,
        inputContext: self.prefillCacheState.inputContext,
        processor: processor,
        sampler: sampler,
        synchronizeStreamForMemorySnapshots: parameters.synchronizeStreamForMemorySnapshots,
        generationStartSnapshot: generationStartSnapshot,
        postPrefillSnapshot: postPrefillSnapshot
      )
      return EdgeToolsGenerationLoop.Preparation(metrics: metrics)
    }

    public nonisolated(nonsending) mutating func decode(
      bitmask: GrammarBitmask,
      parameters: Profile.GenerateParameters
    ) async throws -> EdgeToolsToken.ID {
      guard var generation = self.generation else {
        throw EdgeToolsError.modelNotPrepared
      }
      if let pendingTokenId = generation.pendingTokenId {
        maybeQuantizeKVCache(
          cache: &generation.cache,
          kvBits: parameters.kvCacheQuantizationBits,
          kvGroupSize: parameters.kvCacheQuantizationGroupSize,
          quantizedKVStart: parameters.quantizedKVStart
        )
        let token = MLXArray([pendingTokenId])
        let output = self.languageModel(
          LMInput.Text(tokens: token)[text: .newAxis],
          cache: generation.cache,
          state: generation.outputState
        )
        generation.outputState = output.state
        generation.logits = output.logits
        generation.cachedTokenIds.append(pendingTokenId)
      }
      var stepLogits = generation.logits[0..., -1, 0...]
      stepLogits = generation.processor?.process(logits: stepLogits) ?? stepLogits
      let maskedLogits = applyBitmaskMLX(logits: stepLogits, mask: bitmask)
      let confidenceValues = top(maskedLogits.flattened(), k: 2)
      let token = generation.sampler.sample(logits: maskedLogits)
      eval(confidenceValues, token)

      let confidence = tokenConfidence(unorderedPair: confidenceValues.asArray(Float.self))
      let tokenId = token.item(EdgeToolsToken.ID.self)
      generation.confidence.add(confidence: confidence)
      generation.processor?.didSample(token: token)
      generation.pendingTokenId = tokenId
      self.generation = generation
      return tokenId
    }

    public func finish() -> EdgeToolsMetadata {
      guard let generation = self.generation else { return EdgeToolsMetadata() }
      var metadata = EdgeToolsMetadata()
      metadata.generationConfidence = generation.confidence.mean
      metadata.perTokenConfidences = generation.confidence.perTokenConfidences
      metadata.mlxEngineGenerationStartMemorySnapshot = generation.generationStartSnapshot
      metadata.mlxEnginePostPrefillMemorySnapshot = generation.postPrefillSnapshot
      metadata.mlxEnginePostDecodeMemorySnapshot = Self.memorySnapshot(
        synchronize: generation.synchronizeStreamForMemorySnapshots
      )
      return metadata
    }
  }
#endif
