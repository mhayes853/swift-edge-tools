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

  // MARK: - MLXModelState

  private struct MLXCachedPrefillSnapshot {
    let input: LMInput
    let tokenIds: [EdgeToolsToken.ID]
    let cache: [any KVCache]
    let output: LMOutput
    let context: EdgeToolsLLMPrefillContext?
  }

  public struct MLXModelState<Profile: MLXModelProfile> {
    private final class CachedPrefill {

      let input: LMInput
      let tokenIds: [EdgeToolsToken.ID]
      let cache: [any KVCache]
      let output: LMOutput
      let context: EdgeToolsLLMPrefillContext?
      let inputKind: MLXInputKind
      private let copying = Lock(())

      init(
        input: LMInput,
        tokenIds: [EdgeToolsToken.ID],
        cache: [any KVCache],
        output: LMOutput,
        context: EdgeToolsLLMPrefillContext?,
        inputKind: MLXInputKind
      ) {
        self.input = input
        self.tokenIds = tokenIds
        self.cache = cache
        self.output = output
        self.context = context
        self.inputKind = inputKind
      }

      func input(
        for context: EdgeToolsLLMPrefillContext,
        kind: MLXInputKind
      ) -> LMInput? {
        self.context == context && self.inputKind == kind ? self.input : nil
      }

      func mutableSnapshot(
        tokenIds: [EdgeToolsToken.ID],
        input: LMInput,
        inputContext: EdgeToolsLLMPrefillContext?
      ) -> MLXCachedPrefillSnapshot? {
        guard tokenIds.starts(with: self.tokenIds),
          mlxPrefillContextMatches(
            cachedInput: self.input,
            input: input,
            cachedContext: self.context,
            inputContext: inputContext
          )
        else {
          return nil
        }
        return MLXCachedPrefillSnapshot(
          input: self.input,
          tokenIds: self.tokenIds,
          cache: self.copiedCache(),
          output: self.output,
          context: self.context
        )
      }

      func forked(copyingCache: Bool) -> CachedPrefill {
        guard copyingCache else {
          return self
        }
        let cache = self.copiedCache()
        eval(cache)
        return CachedPrefill(
          input: self.input,
          tokenIds: self.tokenIds,
          cache: cache,
          output: self.output,
          context: self.context,
          inputKind: self.inputKind
        )
      }

      private func copiedCache() -> [any KVCache] {
        // The lock closure runs synchronously and exclusively. The local only escapes after the
        // copy completes, but Swift's isolation checker cannot express that transfer today.
        nonisolated(unsafe) var copiedCache: [any KVCache]?
        self.copying.withLock { _ in
          copiedCache = self.cache.map { $0.copy() }
        }
        return copiedCache!
      }
    }

    package struct PrefillCacheState {
      var cachedPrefill: CachedPrefill?
      var inputContext: EdgeToolsLLMPrefillContext?

      mutating func input(
        for context: EdgeToolsLLMPrefillContext,
        kind: MLXInputKind
      ) -> LMInput? {
        self.inputContext = context
        return self.cachedPrefill?.input(for: context, kind: kind)
      }

      mutating func clearInputContext() {
        self.inputContext = nil
      }

      mutating func fork(copyingCache: Bool) {
        self.cachedPrefill = self.cachedPrefill?.forked(copyingCache: copyingCache)
      }
    }

    package struct Generation {
      let input: LMInput
      var cachedTokenIds: [EdgeToolsToken.ID]
      var cache: [any KVCache]
      var outputState: LMOutput.State?
      var logits: MLXArray
      var pendingTokenId: EdgeToolsToken.ID?
      let inputContext: EdgeToolsLLMPrefillContext?
      var processor: (any LogitProcessor)?
      let sampler: any LogitSampler
      var confidence = ConfidenceState()
      let synchronizeStreamForMemorySnapshots: Bool
      let generationStartSnapshot: Memory.Snapshot
      let postPrefillSnapshot: Memory.Snapshot
    }

    package let vocabularySizeValue: Int
    package var languageModel: any LanguageModel
    package let processor: (any UserInputProcessor)?
    package let configuredExtraStopTokenIds: Set<EdgeToolsToken.ID>
    package let configuredSampling: EdgeToolsFusedSamplingParameters?
    package var prefillCacheState = PrefillCacheState()
    package var generation: Generation?

    public init(
      languageModel: any LanguageModel,
      processor: (any UserInputProcessor)? = nil,
      vocabularySize: Int,
      extraStopTokenIds: Set<EdgeToolsToken.ID>,
      defaultSampling: EdgeToolsFusedSamplingParameters? = nil
    ) {
      self.languageModel = languageModel
      self.processor = processor
      self.vocabularySizeValue = vocabularySize
      self.configuredExtraStopTokenIds = extraStopTokenIds
      self.configuredSampling = defaultSampling
    }

  }
#endif
