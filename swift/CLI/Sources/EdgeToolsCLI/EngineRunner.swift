import EdgeTools
import Foundation
#if canImport(CoreML)
  import CoreML
#endif
#if canImport(MLX)
  import MLX
#endif

// MARK: - EngineRunner

public struct EngineRunner: Sendable {
  public let engine: EngineKind
  public let supportsCustomGrammar: Bool
  public let supportsSampling: Bool
  public let supportsImages: Bool
  public let usesMLX: Bool

  private let generation:
    @Sendable (GenerationRequest, sending EdgeToolsGenerationChannel) async throws ->
      EdgeToolsEngineGeneration
  private let modelResetting: @Sendable () async -> Void

  public init(
    engine: EngineKind = .mlx,
    supportsCustomGrammar: Bool,
    supportsSampling: Bool,
    supportsImages: Bool = false,
    usesMLX: Bool = false,
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
    self.usesMLX = usesMLX
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
    usesMLX: Bool = false,
    prompt: @escaping @Sendable (GenerationRequest) -> Engine.Prompt,
    parameters: @escaping @Sendable (GenerationRequest) throws -> sending Engine.GenerateParameters,
    modelResetting: @escaping @Sendable () async -> Void
  ) {
    self.init(
      engine: engineKind,
      supportsCustomGrammar: supportsCustomGrammar,
      supportsSampling: supportsSampling,
      supportsImages: supportsImages,
      usesMLX: usesMLX,
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
      usesMLX: engineKind == .mlx,
      prompt: prompt,
      parameters: parameters,
      modelResetting: { await engine.resetGeneration() }
    )
  }
}

// MARK: - Loading

extension EngineRunner {
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
      hardwareUnit: hardwareUnit,
      loader: Self.load
    )
  }

  public init(
    detection: ModelDetection,
    engine: EngineKind,
    hardwareUnit: MLXHardwareUnit = .gpu
  ) async throws {
    try await self.init(
      detection: detection,
      engine: try resolvedEngine(requested: engine, detection: detection),
      hardwareUnit: hardwareUnit,
      loader: Self.load
    )
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
      usesMLX: implementation.usesMLX,
      generation: implementation.generation,
      modelResetting: implementation.modelResetting
    )
  }

  private static func load(
    detection: ModelDetection,
    engine: EngineKind,
    hardwareUnit: MLXHardwareUnit
  ) async throws -> Self {
    let directory = detection.directory
    switch (detection.model, engine) {
    #if canImport(MLX)
      case (.needle, .mlx):
        return try await Self.needleMLX(from: directory, hardwareUnit: hardwareUnit)
      case (.qwen3, .mlx), (.genericLLM, .mlx):
        return try await Self.mlx(hardwareUnit: hardwareUnit) {
          try await Qwen3MLXModelEngine(from: directory)
        }
      case (.qwen3P5, .mlx):
        return try await Self.mlx(hardwareUnit: hardwareUnit) {
          try await Qwen3P5MLXModelEngine(from: directory)
        }
      case (.qwen3P5VL, .mlx):
        return try await Self.mlx(hardwareUnit: hardwareUnit, supportsImages: true) {
          try await Qwen3P5VLMLXModelEngine(from: directory)
        }
      case (.lfm2, .mlx):
        return try await Self.mlx(hardwareUnit: hardwareUnit) {
          try await LFM2P5MLXModelEngine(from: directory)
        }
      case (.functionGemma, .mlx):
        return try await Self.mlx(hardwareUnit: hardwareUnit) {
          try await FunctionGemmaMLXModelEngine(from: directory)
        }
      case (.granite, .mlx):
        return try await Self.mlx(hardwareUnit: hardwareUnit) {
          try await GraniteMLXModelEngine(from: directory)
        }
      case (.graniteMoeHybrid, .mlx):
        return try await Self.mlx(hardwareUnit: hardwareUnit) {
          try await GraniteMoeHybridMLXModelEngine(from: directory)
        }
      case (.miniCPM5, .mlx):
        return try await Self.mlx(hardwareUnit: hardwareUnit) {
          try await MiniCPM5MLXModelEngine(from: directory)
        }
      case (.gemma4, .mlx):
        return try await Self.mlx(hardwareUnit: hardwareUnit, supportsImages: true) {
          try await Gemma4MLXModelEngine(from: directory)
        }
      case (.lfm2P5VL, .mlx):
        return try await Self.mlx(hardwareUnit: hardwareUnit, supportsImages: true) {
          try await LFM2P5VLMLXModelEngine(from: directory)
        }
      case (.genericVLM, .mlx):
        return try await Self.mlx(hardwareUnit: hardwareUnit, supportsImages: true) {
          try await GenericVLMMLXModelEngine(from: directory)
        }
    #endif
    case (.needle, .onnx): return try await Self.needleONNX(from: directory)
    #if canImport(CoreML)
      case (.needle, .coreml): return try await Self.needleCoreML(from: directory)
    #endif
    case (.needle, .coreai): return try await Self.needleCoreAI(from: directory)
    default:
      throw EdgeCLIError(
        "\(detection.model.displayName) does not support the \(engine.rawValue) engine."
      )
    }
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

  #if canImport(CoreML)
    private static func needleCoreML(from directory: URL) async throws -> Self {
      let engine = try await NeedleCoreMLModelEngine(
        modelDirectoryURL: directory,
        modelConfiguration: MLModelConfiguration()
      )
      return Self(
        engineKind: .coreml,
        engine: engine,
        prompt: needlePrompt,
        parameters: { request in
          NeedleCoreMLModel.GenerateParameters(
            maxTokens: request.maxTokens,
            toolCallRange: request.toolCallRange
          )
        }
      )
    }
  #endif

  // CoreAI is experimental, needs Swift 6.4 to build, and needs OS 27 to run.
  private static func needleCoreAI(from directory: URL) async throws -> Self {
    #if swift(>=6.4) && canImport(CoreAI)
      guard #available(macOS 27.0, *) else {
        throw EdgeCLIError("The coreai engine needs macOS 27 or newer.")
      }
      let engine = try await NeedleCoreAIModelEngine(modelDirectoryURL: directory)
      return Self(
        engineKind: .coreai,
        engine: engine,
        prompt: needlePrompt,
        parameters: { request in
          NeedleCoreAIModel.GenerateParameters(
            maxTokens: request.maxTokens,
            toolCallRange: request.toolCallRange
          )
        }
      )
    #else
      throw EdgeCLIError("This build of edge has no coreai engine; it requires Swift 6.4 or newer.")
    #endif
  }

  #if canImport(MLX)
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
        usesMLX: true,
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
  if let defaultEngine = available.first(where: { !$0.isExperimental }) {
    return defaultEngine
  }
  let experimental = available.filter(\.isExperimental).map(\.rawValue)
  throw EdgeCLIError(
    """
    No usable engine for \(detection.model.displayName) in \(detection.directory.path()). \
    \(experimental.isEmpty
      ? "Supported engines: \(detection.model.supportedEngines.map(\.rawValue).joined(separator: ", "))."
      : "Select one explicitly with --engine: \(experimental.joined(separator: ", ")).")
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
