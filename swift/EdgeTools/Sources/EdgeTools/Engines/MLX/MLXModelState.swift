#if MLX && canImport(MLX)
  import EdgeToolsCore
  import MLX
  import MLXLMCommon

  // MARK: - MLXContext

  public struct MLXKVCacheQuantization: Hashable, Sendable {
    public var bits: Int
    public var groupSize: Int
    public var startTokenCount: Int

    public init(bits: Int, groupSize: Int = 64, startTokenCount: Int = 0) {
      self.bits = bits
      self.groupSize = groupSize
      self.startTokenCount = startTokenCount
    }
  }

  public struct MLXContextParameters: Hashable, Sendable {
    public var transcript: EdgeToolsTranscript
    public var reasoningEffort: EdgeToolsReasoningEffort
    public var maxKVSize: Int?
    public var prefillChunkSize: Int
    public var kvCacheQuantization: MLXKVCacheQuantization?

    public init(
      transcript: EdgeToolsTranscript = EdgeToolsTranscript(),
      reasoningEffort: EdgeToolsReasoningEffort = .default,
      maxKVSize: Int? = nil,
      prefillChunkSize: Int = 512,
      kvCacheQuantization: MLXKVCacheQuantization? = nil
    ) {
      self.transcript = transcript
      self.reasoningEffort = reasoningEffort
      self.maxKVSize = maxKVSize
      self.prefillChunkSize = Swift.max(prefillChunkSize, 1)
      self.kvCacheQuantization = kvCacheQuantization
    }
  }

  public typealias MLXContext<Profile> = EdgeToolsTranscriptContext<MLXModelState<Profile>>
  where Profile: MLXModelProfile, Profile.Prompt == EdgeToolsTranscript

  struct MLXCachePolicy: Hashable, Sendable {
    var maxKVSize: Int?
    var prefillChunkSize: Int
    var quantization: MLXKVCacheQuantization?

    init(parameters: MLXContextParameters) {
      self.maxKVSize = parameters.maxKVSize
      self.prefillChunkSize = Swift.max(parameters.prefillChunkSize, 1)
      self.quantization = parameters.kvCacheQuantization
    }
  }

  final class MLXPrefixCacheHandle: Sendable {
    let id: UInt64
    private let release: @Sendable (UInt64) -> Void

    init(id: UInt64, release: @escaping @Sendable (UInt64) -> Void) {
      self.id = id
      self.release = release
    }

    deinit {
      self.release(self.id)
    }
  }

  struct MLXPrefillResult: Sendable {
    let prefill: EdgeToolsEnginePrefill
    let checkpoint: MLXPrefixCacheHandle
  }

  struct MLXGenerationResult: Sendable {
    var generation: EdgeToolsEngineGeneration
    let checkpoint: MLXPrefixCacheHandle?
  }

  // MARK: - MLXModelState

  public struct MLXModelState<Profile: MLXModelProfile>: Sendable {
    let policy: MLXCachePolicy
    var checkpoint: MLXPrefixCacheHandle?

    init(policy: MLXCachePolicy, checkpoint: MLXPrefixCacheHandle? = nil) {
      self.policy = policy
      self.checkpoint = checkpoint
    }

    public func forkedContextState() -> sending Self {
      self
    }
  }

  extension MLXModelState: EdgeToolsForkableModelState {}

  struct MLXGeneration {
    var prefix: MLXPrefixState
    var decoder: DecoderState<any LogitSampler>
    var processor: (any LogitProcessor)?
    let synchronizeStreamForMemorySnapshots: Bool
    let generationStartSnapshot: Memory.Snapshot
    let postPrefillSnapshot: Memory.Snapshot
  }
#endif
