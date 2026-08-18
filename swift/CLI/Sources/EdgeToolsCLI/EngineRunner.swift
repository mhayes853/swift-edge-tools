import EdgeTools
import Foundation
import Synchronization

#if canImport(MLX)
  import MLX
#endif

// MARK: - EngineRunner

private struct EngineFactoryContext: Sendable {
  let detection: ModelDetection
  let hardwareUnit: MLXHardwareUnit
}

private typealias ModelEngineFactory = @Sendable (EngineFactoryContext) async throws -> EngineRunner

private struct ModelEngineRegistration: Sendable {
  let factory: ModelEngineFactory
  let supportsImages: Bool

  init(
    factory: @escaping ModelEngineFactory,
    supportsImages: Bool = false
  ) {
    self.factory = factory
    self.supportsImages = supportsImages
  }
}

private struct ModelRegistration: Sendable {
  let engines: [EngineKind: ModelEngineRegistration]
}

public struct EngineRunner: Sendable {
  public let engine: EngineKind
  public let supportsCustomGrammar: Bool
  public let supportsSampling: Bool
  public let supportsImages: Bool

  private let metricsExtractor: any GenerationMetricsExtractor
  private let generation:
    @Sendable (GenerationRequest, sending EdgeToolsGenerationChannel) async throws ->
      EdgeToolsEngineGeneration
  private let modelResetting: @Sendable () async -> Void
  private let modelWarmingUp: @Sendable (GenerationRequest) async throws -> Void

  public init(
    engine: EngineKind = .mlx,
    supportsCustomGrammar: Bool,
    supportsSampling: Bool,
    supportsImages: Bool = false,
    metricsExtractor: any GenerationMetricsExtractor = StandardGenerationMetricsExtractor(),
    generation:
      @escaping @Sendable (
        GenerationRequest, sending EdgeToolsGenerationChannel
      ) async throws -> EdgeToolsEngineGeneration,
    modelResetting: @escaping @Sendable () async -> Void = {},
    modelWarmingUp: @escaping @Sendable (GenerationRequest) async throws -> Void = { _ in }
  ) {
    self.engine = engine
    self.supportsCustomGrammar = supportsCustomGrammar
    self.supportsSampling = supportsSampling
    self.supportsImages = supportsImages
    self.metricsExtractor = metricsExtractor
    self.generation = generation
    self.modelResetting = modelResetting
    self.modelWarmingUp = modelWarmingUp
  }

  public func generate(
    _ request: GenerationRequest,
    channel: sending EdgeToolsGenerationChannel = EdgeToolsGenerationChannel()
  ) async throws -> EdgeToolsEngineGeneration {
    try await self.generation(request, channel)
  }

  public func reset() async {
    await self.modelResetting()
  }

  /// Pays whatever fixed setup an engine defers to its first generation, such as allocating a
  /// llama context, so it lands outside the measured generation.
  public func warmUp(_ request: GenerationRequest) async throws {
    try await self.modelWarmingUp(request)
  }

  public func metrics(from generation: EdgeToolsEngineGeneration) -> CLIGenerationMetrics {
    self.metricsExtractor.extract(from: generation)
  }
}

// MARK: - Engine Adaptation

extension EngineRunner {
  private init<Engine: EdgeToolsEngine>(
    engineKind: EngineKind,
    engine: Engine,
    supportsCustomGrammar: Bool = false,
    supportsSampling: Bool = false,
    supportsImages: Bool = false,
    metricsExtractor: any GenerationMetricsExtractor = StandardGenerationMetricsExtractor(),
    prompt: @escaping @Sendable (GenerationRequest) -> Engine.Prompt,
    parameters: @escaping @Sendable (GenerationRequest) throws -> sending Engine.GenerateParameters,
    modelResetting: @escaping @Sendable () async -> Void = {}
  ) {
    self.init(
      engine: engineKind,
      supportsCustomGrammar: supportsCustomGrammar,
      supportsSampling: supportsSampling,
      supportsImages: supportsImages,
      metricsExtractor: metricsExtractor,
      generation: { request, channel in
        let task = try engine.generate(
          prompt: prompt(request),
          tools: request.tools,
          parameters: try parameters(request),
          channel: channel
        )
        return try await task.value
      },
      modelResetting: modelResetting
    )
  }
}

// MARK: - Loading

extension EngineRunner {
  static func registeredEngines(for model: DetectedModel) -> [EngineKind] {
    EngineKind.allCases.filter {
      $0.isAvailable && Self.modelRegistrations[model]?.engines[$0] != nil
    }
  }

  struct ParsedConfiguration: Sendable {
    let engine: EngineKind
    let hardwareUnit: MLXHardwareUnit
  }

  public struct Capabilities: Hashable, Sendable {
    public var supportsCustomGrammar: Bool
    public var supportsSampling: Bool
    public var supportsImages: Bool
  }

  public static func capabilities(of engine: EngineKind, for model: DetectedModel) -> Capabilities {
    let supportsImages = Self.modelRegistrations[model]?.engines[engine]?.supportsImages ?? false
    switch engine {
    case .mlx:
      return Capabilities(
        supportsCustomGrammar: true,
        supportsSampling: true,
        supportsImages: supportsImages
      )
    case .needle2:
      return Capabilities(
        supportsCustomGrammar: false,
        supportsSampling: false,
        supportsImages: false
      )
    case .llama:
      return Capabilities(
        supportsCustomGrammar: true,
        supportsSampling: true,
        supportsImages: supportsImages
      )
    }
  }

  static func parse(
    _ request: GenerationRequest,
    detection: ModelDetection,
    requestedEngine: EngineKind?,
    requestedHardwareUnit: MLXHardwareUnit?
  ) throws -> ParsedConfiguration {
    let engine = try resolvedEngine(requested: requestedEngine, detection: detection)
    if requestedHardwareUnit != nil, engine != .mlx {
      throw EdgeCLIError("--hardware-unit only applies to the mlx engine.")
    }

    let capabilities = Self.capabilities(of: engine, for: detection.model)
    guard request.sampling.isEmpty || capabilities.supportsSampling else {
      throw EdgeCLIError(
        "\(detection.model.displayName) on \(engine.rawValue) always samples greedily; sampler options do not apply."
      )
    }
    guard request.images.isEmpty || capabilities.supportsImages else {
      throw EdgeCLIError(
        "\(detection.model.displayName) on \(engine.rawValue) takes text only; --image does not apply."
      )
    }
    if !request.images.isEmpty, engine == .llama, detection.model.modality == .vision {
      _ = try multimodalProjector(in: detection)
    }
    guard request.grammar == .auto || capabilities.supportsCustomGrammar else {
      throw EdgeCLIError(
        """
        \(detection.model.displayName) on \(engine.rawValue) only supports `--grammar auto`; its \
        generate parameters expose a tool call range rather than a full generation constraint.
        """
      )
    }
    return ParsedConfiguration(engine: engine, hardwareUnit: requestedHardwareUnit ?? .gpu)
  }

  public init(
    detection: ModelDetection,
    requestedEngine: EngineKind? = nil,
    hardwareUnit: MLXHardwareUnit = .gpu
  ) async throws {
    try await self.init(
      detection: detection,
      engine: try resolvedEngine(requested: requestedEngine, detection: detection),
      hardwareUnit: hardwareUnit
    )
  }

  public init(
    detection: ModelDetection,
    engine: EngineKind,
    hardwareUnit: MLXHardwareUnit = .gpu
  ) async throws {
    let engine = try resolvedEngine(requested: engine, detection: detection)
    let context = EngineFactoryContext(detection: detection, hardwareUnit: hardwareUnit)
    guard let registration = Self.modelRegistrations[detection.model]?.engines[engine] else {
      throw EdgeCLIError(
        "\(detection.model.displayName) does not support the \(engine.rawValue) engine."
      )
    }
    self = try await registration.factory(context)
  }

  public init(
    detection: ModelDetection,
    requestedEngine: EngineKind?,
    hardwareUnit: MLXHardwareUnit,
    loader: @Sendable (ModelDetection, EngineKind, MLXHardwareUnit) async throws -> EngineRunner
  ) async throws {
    try await self.init(
      detection: detection,
      engine: try resolvedEngine(requested: requestedEngine, detection: detection),
      hardwareUnit: hardwareUnit,
      loader: loader
    )
  }

  private init(
    detection: ModelDetection,
    engine: EngineKind,
    hardwareUnit: MLXHardwareUnit,
    loader: @Sendable (ModelDetection, EngineKind, MLXHardwareUnit) async throws -> EngineRunner
  ) async throws {
    let implementation = try await loader(detection, engine, hardwareUnit)
    self.init(
      engine: engine,
      supportsCustomGrammar: implementation.supportsCustomGrammar,
      supportsSampling: implementation.supportsSampling,
      supportsImages: implementation.supportsImages,
      generation: implementation.generation,
      modelResetting: implementation.modelResetting,
      modelWarmingUp: implementation.modelWarmingUp
    )
  }

  private static let modelRegistrations: [DetectedModel: ModelRegistration] = {
    var registrations: [DetectedModel: ModelRegistration] = [:]
    if #available(macOS 26, *) {
      registrations[.needle2] = ModelRegistration(engines: [
        .needle2: ModelEngineRegistration(factory: { _ in Self.needle2() })
      ])
    }
    let llamaRegistrations: [DetectedModel: ModelEngineRegistration] = [
      .qwen3: ModelEngineRegistration(
        factory: Self.llamaFactory { try Qwen3LlamaModelEngine(modelPath: $0) }
      ),
      .qwen3P5: ModelEngineRegistration(
        factory: Self.llamaFactory { try Qwen3P5LlamaModelEngine(modelPath: $0) }
      ),
      .functionGemma: ModelEngineRegistration(
        factory: Self.llamaFactory { try FunctionGemmaLlamaModelEngine(modelPath: $0) }
      ),
      .gemma4: ModelEngineRegistration(
        factory: Self.llamaMultimodalFactory(
          withProjector: {
            try Gemma4LlamaModelEngine(modelPath: $0, multimodalProjectorPath: $1)
          },
          withoutProjector: { try Gemma4LlamaModelEngine(modelPath: $0) }
        ),
        supportsImages: true
      ),
      .lfm2: ModelEngineRegistration(
        factory: Self.llamaFactory { try LFM2P5LlamaModelEngine(modelPath: $0) }
      ),
      .granite: ModelEngineRegistration(
        factory: Self.llamaFactory { try GraniteLlamaModelEngine(modelPath: $0) }
      ),
      .graniteMoeHybrid: ModelEngineRegistration(
        factory: Self.llamaFactory { try GraniteLlamaModelEngine(modelPath: $0) }
      ),
      .miniCPM5: ModelEngineRegistration(
        factory: Self.llamaFactory { try MiniCPM5LlamaModelEngine(modelPath: $0) }
      ),
      .genericLLM: ModelEngineRegistration(
        factory: Self.llamaFactory { try Qwen3LlamaModelEngine(modelPath: $0) }
      ),
      .genericVLM: ModelEngineRegistration(
        factory: Self.llamaMultimodalFactory(
          withProjector: {
            try GenericVLMLlamaModelEngine(modelPath: $0, multimodalProjectorPath: $1)
          },
          withoutProjector: { try GenericVLMLlamaModelEngine(modelPath: $0) }
        ),
        supportsImages: true
      )
    ]
    #if canImport(MLX)
      registrations[.qwen3] = ModelRegistration(engines: [
        .mlx: ModelEngineRegistration(
          factory: Self.mlxFactory { try await Qwen3MLXModelEngine(from: $0) }
        )
      ])
      registrations[.qwen3P5] = ModelRegistration(engines: [
        .mlx: ModelEngineRegistration(
          factory: Self.mlxFactory { try await Qwen3P5MLXModelEngine(from: $0) }
        )
      ])
      registrations[.qwen3P5VL] = ModelRegistration(engines: [
        .mlx: ModelEngineRegistration(
          factory: Self.mlxFactory(
            { try await Qwen3P5VLMLXModelEngine(from: $0) },
            supportsImages: true
          ),
          supportsImages: true
        )
      ])
      registrations[.lfm2] = ModelRegistration(engines: [
        .mlx: ModelEngineRegistration(
          factory: Self.mlxFactory { try await LFM2P5MLXModelEngine(from: $0) }
        )
      ])
      registrations[.functionGemma] = ModelRegistration(engines: [
        .mlx: ModelEngineRegistration(
          factory: Self.mlxFactory { try await FunctionGemmaMLXModelEngine(from: $0) }
        )
      ])
      registrations[.gemma4] = ModelRegistration(engines: [
        .mlx: ModelEngineRegistration(
          factory: Self.mlxFactory(
            { try await Gemma4MLXModelEngine(from: $0) },
            supportsImages: true
          ),
          supportsImages: true
        )
      ])
      registrations[.granite] = ModelRegistration(engines: [
        .mlx: ModelEngineRegistration(
          factory: Self.mlxFactory { try await GraniteMLXModelEngine(from: $0) }
        )
      ])
      registrations[.graniteMoeHybrid] = ModelRegistration(engines: [
        .mlx: ModelEngineRegistration(
          factory: Self.mlxFactory { try await GraniteMoeHybridMLXModelEngine(from: $0) }
        )
      ])
      registrations[.miniCPM5] = ModelRegistration(engines: [
        .mlx: ModelEngineRegistration(
          factory: Self.mlxFactory { try await MiniCPM5MLXModelEngine(from: $0) }
        )
      ])
      registrations[.lfm2P5VL] = ModelRegistration(engines: [
        .mlx: ModelEngineRegistration(
          factory: Self.mlxFactory(
            { try await LFM2P5VLMLXModelEngine(from: $0) },
            supportsImages: true
          ),
          supportsImages: true
        )
      ])
      registrations[.genericLLM] = ModelRegistration(engines: [
        .mlx: ModelEngineRegistration(
          factory: Self.mlxFactory { try await Qwen3MLXModelEngine(from: $0) }
        )
      ])
      registrations[.genericVLM] = ModelRegistration(engines: [
        .mlx: ModelEngineRegistration(
          factory: Self.mlxFactory(
            { try await GenericVLMMLXModelEngine(from: $0) },
            supportsImages: true
          ),
          supportsImages: true
        )
      ])
    #endif
    for (model, registration) in llamaRegistrations {
      var engines = registrations[model]?.engines ?? [:]
      engines[.llama] = registration
      registrations[model] = ModelRegistration(engines: engines)
    }
    return registrations
  }()

  @available(macOS 26, *)
  private static func needle2() -> Self {
    let engine = Needle2Engine()
    return Self(
      engine: .needle2,
      supportsCustomGrammar: false,
      supportsSampling: false,
      metricsExtractor: Needle2GenerationMetricsExtractor(),
      generation: { request, channel in
        let context = engine.context(
          Needle2ContextParameters(system: try needle2System(from: request.system))
        )
        let task = try engine.generate(
          prompt: request.user,
          tools: request.tools,
          parameters: Needle2GenerateParameters(maxTokens: request.maxTokens),
          context: context,
          channel: channel
        )
        return try await task.value
      }
    )
  }

  private static func llamaFactory<Profile: LlamaModelProfile>(
    _ make: @escaping @Sendable (String) throws -> LlamaEngine<Profile>
  ) -> ModelEngineFactory
  where
    Profile.Prompt == EdgeToolsTranscript,
    Profile.GenerateParameters == DefaultLlamaGenerateParameters,
    Profile.GrammarEngine == XGrammarEngine
  {
    { context in
      guard let file = context.detection.ggufFile else {
        throw EdgeCLIError(
          """
          No GGUF file in \(context.detection.directory.path()) for the llama engine. Download a \
          GGUF build of \(context.detection.model.displayName), or pick one with --quant.
          """
        )
      }
      return Self.llama(engine: try make(file.path()))
    }
  }

  private static func llamaMultimodalFactory<Profile: LlamaModelProfile & EdgeToolsMultimodalModelProfile>(
    withProjector: @escaping @Sendable (String, String) throws -> LlamaEngine<Profile>,
    withoutProjector: @escaping @Sendable (String) throws -> LlamaEngine<Profile>
  ) -> ModelEngineFactory
  where
    Profile.Prompt == EdgeToolsTranscript,
    Profile.GenerateParameters == DefaultLlamaGenerateParameters,
    Profile.GrammarEngine == XGrammarEngine
  {
    { context in
      guard let modelFile = context.detection.ggufFile else {
        throw EdgeCLIError(
          "No GGUF file in \(context.detection.directory.path()) for the llama engine."
        )
      }
      guard !context.detection.multimodalProjectorFiles.isEmpty else {
        return Self.llama(engine: try withoutProjector(modelFile.path()))
      }
      let projector = try multimodalProjector(in: context.detection)
      return Self.llama(
        engine: try withProjector(modelFile.path(), projector.path()),
        supportsImages: true
      )
    }
  }

  private static func llama<Profile: LlamaModelProfile>(
    engine: LlamaEngine<Profile>,
    supportsImages: Bool = false
  ) -> Self
  where
    Profile.Prompt == EdgeToolsTranscript,
    Profile.GenerateParameters == DefaultLlamaGenerateParameters,
    Profile.GrammarEngine == XGrammarEngine
  {
    // Held across generations so multi-turn runs reuse the KV cache; `bench` drops it per run.
    let cachedContext = Mutex<LlamaContext<Profile>?>(nil)
    return Self(
      engine: .llama,
      supportsCustomGrammar: true,
      supportsSampling: true,
      supportsImages: supportsImages,
      generation: { request, channel in
        let context = llamaContext(engine: engine, cache: cachedContext, request: request)
        let task = try engine.generate(
          prompt: .user(request.user, images: request.images),
          tools: request.tools,
          parameters: DefaultLlamaGenerateParameters(
            sampling: request.sampling,
            constraint: try request.grammar.constraint(toolCallRange: request.toolCallRange),
            maxTokens: request.maxTokens
          ),
          context: context,
          channel: channel
        )
        return try await task.value
      },
      modelResetting: { cachedContext.withLock { $0 = nil } },
      modelWarmingUp: { request in
        try engine.warmUp(
          tools: request.tools,
          context: llamaContext(engine: engine, cache: cachedContext, request: request)
        )
      }
    )
  }

  #if canImport(MLX)
    private static func mlxFactory<Profile: MLXModelProfile>(
      _ make: @escaping @Sendable (URL) async throws -> MLXEngine<Profile>,
      supportsImages: Bool = false
    ) -> ModelEngineFactory
    where
      Profile.Prompt == EdgeToolsTranscript,
      Profile.GenerateParameters == DefaultMLXGenerateParameters,
      Profile.GrammarEngine == XGrammarEngine
    {
      { context in
        try await Self.mlx(
          hardwareUnit: context.hardwareUnit,
          supportsImages: supportsImages
        ) {
          try await make(context.detection.directory)
        }
      }
    }

    private static func mlx<Profile: MLXModelProfile>(
      hardwareUnit: MLXHardwareUnit,
      supportsImages: Bool = false,
      make: () async throws -> MLXEngine<Profile>
    ) async throws -> Self
    where
      Profile.Prompt == EdgeToolsTranscript,
      Profile.GenerateParameters == DefaultMLXGenerateParameters,
      Profile.GrammarEngine == XGrammarEngine
    {
      let engine = try await Device.withDefaultDevice(hardwareUnit.device) {
        try await make()
      }
      return Self(
        engine: .mlx,
        supportsCustomGrammar: true,
        supportsSampling: true,
        supportsImages: supportsImages,
        generation: { request, channel in
          let context = engine.context(
            MLXContextParameters(
              transcript: EdgeToolsTranscript(
                messages: request.system.isEmpty ? [] : [.system(request.system)]
              ),
              reasoningEffort: request.reasoning
            )
          )
          let task = try engine.generate(
            prompt: .user(
              request.user,
              images: request.images
            ),
            tools: request.tools,
            parameters: DefaultMLXGenerateParameters(
              sampling: request.sampling,
              constraint: try request.grammar.constraint(
                toolCallRange: request.toolCallRange
              ),
              maxTokens: request.maxTokens
            ),
            context: context,
            channel: channel
          )
          return try await task.value
        },
        modelResetting: {}
      )
    }
  #endif
}

private func needle2System(from string: String) throws -> Needle2System {
  guard !string.isEmpty else { return [] }
  return try Needle2System(
    string.split(separator: ";")
      .map { fact in
        let components = fact.split(separator: ":", maxSplits: 1)
        guard components.count == 2 else {
          throw EdgeCLIError(
            "Needle 2 system facts must use `key: value` entries separated by semicolons."
          )
        }
        let key = components[0].trimmingCharacters(in: .whitespacesAndNewlines)
        let value = components[1].trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty, !value.isEmpty else {
          throw EdgeCLIError(
            "Needle 2 system facts must use non-empty keys and values."
          )
        }
        return Needle2System.raw(.init(rawValue: key), value)
      }
  )
}

private func llamaContext<Profile: LlamaModelProfile>(
  engine: LlamaEngine<Profile>,
  cache: borrowing Mutex<LlamaContext<Profile>?>,
  request: GenerationRequest
) -> LlamaContext<Profile> {
  cache.withLock { context in
    if let context {
      return context
    }
    let created = engine.context(
      EdgeToolsTranscriptContextParameters(
        transcript: EdgeToolsTranscript(
          messages: request.system.isEmpty ? [] : [.system(request.system)]
        ),
        reasoningEffort: request.reasoning
      )
    )
    context = created
    return created
  }
}

private func resolvedEngine(
  requested: EngineKind?,
  detection: ModelDetection
) throws -> EngineKind {
  let available = detection.engines.filter(detection.model.supportedEngines.contains)
  if let requested {
    guard available.contains(requested) else {
      throw EdgeCLIError(
        """
        The \(requested.rawValue) engine has no weights for \
        \(detection.model.displayName) here. Available: \
        \(available.map(\.rawValue).joined(separator: ", ")).
        """
      )
    }
    return requested
  }
  if let defaultEngine = detection.defaultEngine {
    return defaultEngine
  }
  throw EdgeCLIError(
    """
    No usable engine for \(detection.model.displayName) in \(detection.directory.path()). \
    Supported engines: \(detection.model.supportedEngines.map(\.rawValue).joined(separator: ", ")).
    """
  )
}

private func multimodalProjector(in detection: ModelDetection) throws -> URL {
  let projectors = detection.multimodalProjectorFiles
  guard projectors.count == 1, let projector = projectors.first else {
    let files = projectors.map(\.lastPathComponent).joined(separator: " · ")
    throw EdgeCLIError(
      projectors.isEmpty
        ? "No multimodal projector GGUF in \(detection.directory.path()). Expected an mmproj*.gguf file."
        : "Several multimodal projector GGUF files in \(detection.directory.path()): \(files)."
      )
  }
  return projector
}
