import EdgeTools
import Foundation

#if canImport(MLX)
  import MLX
#endif

// MARK: - EngineRunner

private struct EngineFactoryContext: Sendable {
  let detection: ModelDetection
  let hardwareUnit: MLXHardwareUnit
}

private typealias ModelEngineFactory = @Sendable (EngineFactoryContext) async throws -> EngineRunner

private struct ModelRegistration: Sendable {
  let engines: [EngineKind: ModelEngineFactory]
}

public struct EngineRunner: Sendable {
  public let engine: EngineKind
  public let supportsCustomGrammar: Bool
  public let supportsSampling: Bool
  public let supportsImages: Bool

  private let generation:
    @Sendable (GenerationRequest, sending EdgeToolsGenerationChannel) async throws ->
      EdgeToolsEngineGeneration
  private let modelResetting: @Sendable () async -> Void

  public init(
    engine: EngineKind = .mlx,
    supportsCustomGrammar: Bool,
    supportsSampling: Bool,
    supportsImages: Bool = false,
    generation:
      @escaping @Sendable (
        GenerationRequest, sending EdgeToolsGenerationChannel
      ) async throws -> EdgeToolsEngineGeneration,
    modelResetting: @escaping @Sendable () async -> Void = {}
  ) {
    self.engine = engine
    self.supportsCustomGrammar = supportsCustomGrammar
    self.supportsSampling = supportsSampling
    self.supportsImages = supportsImages
    self.generation = generation
    self.modelResetting = modelResetting
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
}

// MARK: - Engine Adaptation

extension EngineRunner {
  private init<Engine: EdgeToolsEngine>(
    engineKind: EngineKind,
    engine: Engine,
    supportsCustomGrammar: Bool,
    supportsSampling: Bool,
    supportsImages: Bool = false,
    prompt: @escaping @Sendable (GenerationRequest) -> Engine.Prompt,
    parameters: @escaping @Sendable (GenerationRequest) throws -> sending Engine.GenerateParameters,
    modelResetting: @escaping @Sendable () async -> Void
  ) {
    self.init(
      engine: engineKind,
      supportsCustomGrammar: supportsCustomGrammar,
      supportsSampling: supportsSampling,
      supportsImages: supportsImages,
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

extension EngineRunner {
  private init<Model: EdgeToolsModel>(
    engineKind: EngineKind,
    engine: EdgeToolsModelEngine<Model>,
    supportsCustomGrammar: Bool = false,
    supportsSampling: Bool = false,
    supportsImages: Bool = false,
    prompt: @escaping @Sendable (GenerationRequest) -> Model.Prompt,
    parameters: @escaping @Sendable (GenerationRequest) throws -> sending Model.GenerateParameters
  ) {
    self.init(
      engineKind: engineKind,
      engine: engine,
      supportsCustomGrammar: supportsCustomGrammar,
      supportsSampling: supportsSampling,
      supportsImages: supportsImages,
      prompt: prompt,
      parameters: parameters,
      modelResetting: { await engine.resetGeneration() }
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

    let supportsMLXFeatures = engine == .mlx && detection.model != .needle
    guard request.sampling.isEmpty || supportsMLXFeatures else {
      throw EdgeCLIError(
        "\(detection.model.displayName) on \(engine.rawValue) always samples greedily; sampler options do not apply."
      )
    }
    let supportsImages = supportsMLXFeatures && detection.model.modality == .vision
    guard request.images.isEmpty || supportsImages else {
      throw EdgeCLIError(
        "\(detection.model.displayName) on \(engine.rawValue) takes text only; --image does not apply."
      )
    }
    guard request.grammar == .auto || supportsMLXFeatures else {
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
    guard let factory = Self.modelRegistrations[detection.model]?.engines[engine] else {
      throw EdgeCLIError(
        "\(detection.model.displayName) does not support the \(engine.rawValue) engine."
      )
    }
    self = try await factory(context)
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
      modelResetting: implementation.modelResetting
    )
  }

  private static let modelRegistrations: [DetectedModel: ModelRegistration] = {
    var registrations: [DetectedModel: ModelRegistration] = [
      .needle: ModelRegistration(engines: Self.needleFactories)
    ]
    #if canImport(MLX)
      registrations[.qwen3] = ModelRegistration(engines: [
        .mlx: Self.mlxFactory { try await Qwen3MLXModelEngine(from: $0) }
      ])
      registrations[.qwen3P5] = ModelRegistration(engines: [
        .mlx: Self.mlxFactory { try await Qwen3P5MLXModelEngine(from: $0) }
      ])
      registrations[.qwen3P5VL] = ModelRegistration(engines: [
        .mlx: Self.mlxFactory(
          { try await Qwen3P5VLMLXModelEngine(from: $0) },
          supportsImages: true
        )
      ])
      registrations[.lfm2] = ModelRegistration(engines: [
        .mlx: Self.mlxFactory { try await LFM2P5MLXModelEngine(from: $0) }
      ])
      registrations[.functionGemma] = ModelRegistration(engines: [
        .mlx: Self.mlxFactory { try await FunctionGemmaMLXModelEngine(from: $0) }
      ])
      registrations[.gemma4] = ModelRegistration(engines: [
        .mlx: Self.mlxFactory(
          { try await Gemma4MLXModelEngine(from: $0) },
          supportsImages: true
        )
      ])
      registrations[.granite] = ModelRegistration(engines: [
        .mlx: Self.mlxFactory { try await GraniteMLXModelEngine(from: $0) }
      ])
      registrations[.graniteMoeHybrid] = ModelRegistration(engines: [
        .mlx: Self.mlxFactory { try await GraniteMoeHybridMLXModelEngine(from: $0) }
      ])
      registrations[.miniCPM5] = ModelRegistration(engines: [
        .mlx: Self.mlxFactory { try await MiniCPM5MLXModelEngine(from: $0) }
      ])
      registrations[.lfm2P5VL] = ModelRegistration(engines: [
        .mlx: Self.mlxFactory(
          { try await LFM2P5VLMLXModelEngine(from: $0) },
          supportsImages: true
        )
      ])
      registrations[.genericLLM] = ModelRegistration(engines: [
        .mlx: Self.mlxFactory { try await Qwen3MLXModelEngine(from: $0) }
      ])
      registrations[.genericVLM] = ModelRegistration(engines: [
        .mlx: Self.mlxFactory(
          { try await GenericVLMMLXModelEngine(from: $0) },
          supportsImages: true
        )
      ])
    #endif
    return registrations
  }()

  private static var needleFactories: [EngineKind: ModelEngineFactory] {
    var factories: [EngineKind: ModelEngineFactory] = [
      .onnx: { context in try await Self.needleONNX(from: context.detection.directory) }
    ]
    #if canImport(MLX)
      factories[.mlx] = { context in
        try await Self.needleMLX(
          from: context.detection.directory,
          hardwareUnit: context.hardwareUnit
        )
      }
    #endif
    return factories
  }

  #if canImport(MLX)
    private static func needleMLX(
      from directory: URL,
      hardwareUnit: MLXHardwareUnit
    ) async throws -> Self {
      let engine = try await Device.withDefaultDevice(hardwareUnit.device) {
        try await NeedleMLXModelEngine(from: directory)
      }
      return Self(
        engineKind: .mlx,
        engine: engine,
        prompt: needlePrompt,
        parameters: { request in
          NeedleMLXGenerateParameters(
            maxTokens: request.maxTokens,
            toolCallRange: request.toolCallRange
          )
        }
      )
    }
  #endif

  private static func needleONNX(from directory: URL) async throws -> Self {
    let engine = try await NeedleCONNXModelEngine(from: directory)
    return Self(
      engineKind: .onnx,
      engine: engine,
      prompt: needlePrompt,
      parameters: { request in
        NeedleONNXGenerateParameters(
          maxTokens: request.maxTokens,
          toolCallRange: request.toolCallRange
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
      Profile.Prompt == EdgeToolsConversationalPrompt,
      Profile.GenerateParameters == DefaultMLXGenerateParameters,
      Profile.GrammarCompiler == XGRCompiler,
      Profile.GrammarContext == XGRGrammarContext
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
      Profile.Prompt == EdgeToolsConversationalPrompt,
      Profile.GenerateParameters == DefaultMLXGenerateParameters,
      Profile.GrammarCompiler == XGRCompiler,
      Profile.GrammarContext == XGRGrammarContext
    {
      let engine = try await Device.withDefaultDevice(hardwareUnit.device) {
        try await make()
      }
      return Self(
        engineKind: .mlx,
        engine: engine,
        supportsCustomGrammar: true,
        supportsSampling: true,
        supportsImages: supportsImages,
        prompt: llmPrompt,
        parameters: { request in
          DefaultMLXGenerateParameters(
            samplingOverrides: request.sampling,
            constraint: try request.grammar.constraint(toolCallRange: request.toolCallRange),
            maxTokens: request.maxTokens
          )
        },
        modelResetting: { await engine.resetGeneration() }
      )
    }
  #endif
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

private func needlePrompt(for request: GenerationRequest) -> NeedlePrompt {
  NeedlePrompt(system: request.system, user: request.user)
}

#if canImport(MLX)
  private func llmPrompt(for request: GenerationRequest) -> EdgeToolsConversationalPrompt {
    var messages = [EdgeToolsConversationalPrompt.Message]()
    if !request.system.isEmpty { messages.append(.system(request.system)) }
    messages.append(.user(request.user, images: request.images))
    return EdgeToolsConversationalPrompt(
      messages: messages,
      reasoningEffort: request.reasoning
    )
  }

#endif
