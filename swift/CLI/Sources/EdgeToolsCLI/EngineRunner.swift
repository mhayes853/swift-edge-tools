import EdgeTools
import Foundation
import Synchronization

#if canImport(MLX)
  import MLX
#endif

// MARK: - EngineCapabilities

public struct EngineCapabilities: OptionSet, Hashable, Sendable {
  public let rawValue: UInt64

  public init(rawValue: UInt64) {
    self.rawValue = rawValue
  }

  public static let customGrammar = Self(rawValue: 1 << 0)
  public static let sampling = Self(rawValue: 1 << 1)
  public static let imageInput = Self(rawValue: 1 << 2)
  public static let audioInput = Self(rawValue: 1 << 3)
}

// MARK: - EngineRunner

private struct EngineFactoryContext: Sendable {
  let detection: ModelDetection
  let hardwareUnit: MLXHardwareUnit
}

private typealias ModelEngineFactory = @Sendable (EngineFactoryContext) async throws -> EngineRunner

private struct ModelEngineRegistration: Sendable {
  let factory: ModelEngineFactory
  let capabilities: EngineCapabilities

  init(
    factory: @escaping ModelEngineFactory,
    capabilities: EngineCapabilities = []
  ) {
    self.factory = factory
    self.capabilities = capabilities
  }
}

private struct ModelRegistration: Sendable {
  let engines: [EngineKind: ModelEngineRegistration]
}

public struct EngineRunner: Sendable {
  public let engine: EngineKind
  public let capabilities: EngineCapabilities

  private let metricsExtractor: any GenerationMetricsExtractor
  private let generation:
    @Sendable (GenerationRequest, sending EdgeToolsGenerationChannel) async throws ->
      EdgeToolsEngineGeneration
  private let modelResetting: @Sendable () async -> Void
  private let modelWarmingUp: @Sendable (GenerationRequest) async throws -> Void

  public init(
    engine: EngineKind = .mlx,
    capabilities: EngineCapabilities = [],
    metricsExtractor: any GenerationMetricsExtractor = StandardGenerationMetricsExtractor(),
    generation:
      @escaping @Sendable (
        GenerationRequest, sending EdgeToolsGenerationChannel
      ) async throws -> EdgeToolsEngineGeneration,
    modelResetting: @escaping @Sendable () async -> Void = {},
    modelWarmingUp: @escaping @Sendable (GenerationRequest) async throws -> Void = { _ in }
  ) {
    self.engine = engine
    self.capabilities = capabilities
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
    capabilities: EngineCapabilities = [],
    metricsExtractor: any GenerationMetricsExtractor = StandardGenerationMetricsExtractor(),
    prompt: @escaping @Sendable (GenerationRequest) -> Engine.Prompt,
    parameters: @escaping @Sendable (GenerationRequest) throws -> sending Engine.GenerateParameters,
    modelResetting: @escaping @Sendable () async -> Void = {}
  ) {
    self.init(
      engine: engineKind,
      capabilities: capabilities,
      metricsExtractor: metricsExtractor,
      generation: { request, channel in
        let context = engine.context(tools: definitionTools(request.tools))
        let task = try engine.generate(
          prompt: prompt(request),
          parameters: try parameters(request),
          context: context,
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

  public static func capabilities(
    of engine: EngineKind,
    for model: DetectedModel
  ) -> EngineCapabilities {
    let engineCapabilities: EngineCapabilities =
      switch engine {
      case .mlx, .llama: [.customGrammar, .sampling]
      case .needle2: []
      }
    let modelCapabilities = Self.modelRegistrations[model]?.engines[engine]?.capabilities ?? []
    return engineCapabilities.union(modelCapabilities)
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
    guard request.sampling.isEmpty || capabilities.contains(.sampling) else {
      throw EdgeCLIError(
        "\(detection.model.displayName) on \(engine.rawValue) always samples greedily; sampler options do not apply."
      )
    }
    guard request.images.isEmpty || capabilities.contains(.imageInput) else {
      throw EdgeCLIError(
        "\(detection.model.displayName) on \(engine.rawValue) does not support --image."
      )
    }
    guard request.audio.isEmpty || capabilities.contains(.audioInput) else {
      throw EdgeCLIError(
        "\(detection.model.displayName) on \(engine.rawValue) does not support --audio."
      )
    }
    if engine == .llama, !request.images.isEmpty || !request.audio.isEmpty {
      _ = try multimodalProjector(in: detection)
    }
    guard request.grammar == .auto || capabilities.contains(.customGrammar) else {
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
    let implementation = try await registration.factory(context)
    self = implementation.withCapabilities(Self.capabilities(of: engine, for: detection.model))
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
      capabilities: implementation.capabilities,
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
        capabilities: [.imageInput, .audioInput]
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
        capabilities: [.imageInput, .audioInput]
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
          factory: Self.mlxFactory { try await Qwen3P5VLMLXModelEngine(from: $0) },
          capabilities: [.imageInput]
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
          factory: Self.mlxFactory { try await Gemma4MLXModelEngine(from: $0) },
          capabilities: [.imageInput]
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
          factory: Self.mlxFactory { try await LFM2P5VLMLXModelEngine(from: $0) },
          capabilities: [.imageInput]
        )
      ])
      registrations[.genericLLM] = ModelRegistration(engines: [
        .mlx: ModelEngineRegistration(
          factory: Self.mlxFactory { try await Qwen3MLXModelEngine(from: $0) }
        )
      ])
      registrations[.genericVLM] = ModelRegistration(engines: [
        .mlx: ModelEngineRegistration(
          factory: Self.mlxFactory { try await GenericVLMMLXModelEngine(from: $0) },
          capabilities: [.imageInput]
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
      metricsExtractor: Needle2GenerationMetricsExtractor(),
      generation: { request, channel in
        let context = engine.context(
          Needle2ContextParameters(system: try needle2System(from: request.system)),
          tools: definitionTools(request.tools)
        )
        let task = try engine.generate(
          prompt: .user(request.user),
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

  private static func llamaMultimodalFactory<
    Profile: LlamaModelProfile & EdgeToolsMultimodalModelProfile
  >(
    withProjector: @escaping @Sendable (String, String) throws -> LlamaEngine<Profile>,
    withoutProjector: @escaping @Sendable (String) throws -> LlamaEngine<Profile>
  ) -> ModelEngineFactory
  where
    Profile.Prompt == EdgeToolsTranscript,
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
      return Self.llama(engine: try withProjector(modelFile.path(), projector.path()))
    }
  }

  private static func llama<Profile: LlamaModelProfile>(
    engine: LlamaEngine<Profile>
  ) -> Self
  where
    Profile.Prompt == EdgeToolsTranscript,
    Profile.GrammarEngine == XGrammarEngine
  {
    // Held across generations so multi-turn runs reuse the KV cache; `bench` drops it per run.
    let cachedContext = Mutex<LlamaContext?>(nil)
    return Self(
      engine: .llama,
      capabilities: [.customGrammar, .sampling],
      generation: { request, channel in
        let context = llamaContext(engine: engine, cache: cachedContext, request: request)
        let task = try engine.generate(
          prompt: .user(request.user, images: request.images, audio: request.audio),
          parameters: LlamaGenerateParameters(
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
        _ = try await engine.prefill(
          promptPrefix: EdgeToolsTranscript.Prompt(messages: []),
          context: llamaContext(engine: engine, cache: cachedContext, request: request)
        )
      }
    )
  }

  #if canImport(MLX)
    private static func mlxFactory<Profile: MLXModelProfile>(
      _ make: @escaping @Sendable (URL) async throws -> MLXEngine<Profile>
    ) -> ModelEngineFactory
    where
      Profile.Prompt == EdgeToolsTranscript,
      Profile.GrammarEngine == XGrammarEngine
    {
      { context in
        try await Self.mlx(hardwareUnit: context.hardwareUnit) {
          try await make(context.detection.directory)
        }
      }
    }

    private static func mlx<Profile: MLXModelProfile>(
      hardwareUnit: MLXHardwareUnit,
      make: () async throws -> MLXEngine<Profile>
    ) async throws -> Self
    where
      Profile.Prompt == EdgeToolsTranscript,
      Profile.GrammarEngine == XGrammarEngine
    {
      let engine = try await Device.withDefaultDevice(hardwareUnit.device) {
        try await make()
      }
      return Self(
        engine: .mlx,
        capabilities: [.customGrammar, .sampling],
        generation: { request, channel in
          let context = engine.context(
            MLXContextParameters(
              transcript: EdgeToolsTranscript(
                messages: request.system.isEmpty ? [] : [.system(request.system)]
              ),
              reasoningEffort: request.reasoning
            ),
            tools: definitionTools(request.tools)
          )
          let task = try engine.generate(
            prompt: .user(
              request.user,
              images: request.images,
              audio: request.audio
            ),
            parameters: MLXGenerateParameters(
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
  cache: borrowing Mutex<LlamaContext?>,
  request: GenerationRequest
) -> LlamaContext {
  cache.withLock { context in
    if let context,
      toolDefinitions(context.tools) == request.tools
    {
      return context
    }
    let created = engine.context(
      transcript: EdgeToolsTranscript(
        messages: request.system.isEmpty ? [] : [.system(request.system)]
      ),
      reasoningEffort: request.reasoning,
      tools: definitionTools(request.tools)
    )
    context = created
    return created
  }
}

private struct DefinitionTool: EdgeTool {
  typealias Input = EdgeToolsValue
  typealias Output = EdgeToolsValue

  let definition: EdgeToolDefinition

  var name: String { self.definition.name }
  var description: String { self.definition.description }
  var arguments: EdgeToolsGenerationSchema { self.definition.arguments }
  var includesSchemaInInstructions: Bool { self.definition.includesSchemaInInstructions }

  func invoke(input: EdgeToolsValue) async throws -> EdgeToolsValue {
    input
  }
}

private func definitionTools(_ definitions: [EdgeToolDefinition]) -> [any EdgeTool] {
  definitions.map { DefinitionTool(definition: $0) as any EdgeTool }
}

private func toolDefinitions(_ tools: [any EdgeTool]) -> [EdgeToolDefinition] {
  tools.map { $0.definition }
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

// MARK: - Capability Adaptation

extension EngineRunner {
  private func withCapabilities(_ capabilities: EngineCapabilities) -> Self {
    Self(
      engine: self.engine,
      capabilities: capabilities,
      metricsExtractor: self.metricsExtractor,
      generation: self.generation,
      modelResetting: self.modelResetting,
      modelWarmingUp: self.modelWarmingUp
    )
  }
}
