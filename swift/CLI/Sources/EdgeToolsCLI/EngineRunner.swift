import CoreML
import EdgeTools
import Foundation
import MLX
import MLXLMCommon

// MARK: - EngineRunner

public struct EngineRunner: Sendable {
  public let engine: EngineKind
  public let supportsCustomGrammar: Bool
  public let supportsSampling: Bool
  public let usesMLX: Bool

  private let generation:
    @Sendable (GenerationRequest, sending EdgeToolsGenerationChannel) async throws ->
      EdgeToolsEngineGeneration
  private let cacheClearing: @Sendable () async -> Void

  public init(
    engine: EngineKind = .mlx,
    supportsCustomGrammar: Bool,
    supportsSampling: Bool,
    usesMLX: Bool = false,
    generation:
      @escaping @Sendable (
        GenerationRequest, sending EdgeToolsGenerationChannel
      ) async throws -> EdgeToolsEngineGeneration,
    cacheClearing: @escaping @Sendable () async -> Void = {}
  ) {
    self.engine = engine
    self.supportsCustomGrammar = supportsCustomGrammar
    self.supportsSampling = supportsSampling
    self.usesMLX = usesMLX
    self.generation = generation
    self.cacheClearing = cacheClearing
  }

  public func generate(
    _ request: GenerationRequest,
    channel: sending EdgeToolsGenerationChannel = EdgeToolsGenerationChannel()
  ) async throws -> EdgeToolsEngineGeneration {
    try await self.generation(request, channel)
  }

  public func reset() async {
    await self.cacheClearing()
  }
}

// MARK: - Engine Adaptation

extension EngineRunner {
  private init<Engine: EdgeToolsEngine>(
    engineKind: EngineKind,
    engine: Engine,
    supportsCustomGrammar: Bool,
    supportsSampling: Bool,
    usesMLX: Bool = false,
    prompt: @escaping @Sendable (GenerationRequest) -> Engine.Prompt,
    parameters: @escaping @Sendable (GenerationRequest) throws -> sending Engine.GenerateParameters,
    cacheClearing: @escaping @Sendable () async -> Void
  ) {
    self.init(
      engine: engineKind,
      supportsCustomGrammar: supportsCustomGrammar,
      supportsSampling: supportsSampling,
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
      cacheClearing: cacheClearing
    )
  }
}

// MARK: - Loading

extension EngineRunner {
  public init(
    detection: ModelDetection,
    requestedEngine: EngineKind? = nil,
    hardwareUnit: MLXHardwareUnit = .gpu
  ) async throws {
    try await self.init(
      detection: detection,
      requestedEngine: requestedEngine,
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
    let engine = try resolvedEngine(requested: requestedEngine, detection: detection)
    let implementation = try await loader(detection, engine, hardwareUnit)
    self.init(
      engine: engine,
      supportsCustomGrammar: implementation.supportsCustomGrammar,
      supportsSampling: implementation.supportsSampling,
      usesMLX: implementation.usesMLX,
      generation: implementation.generation,
      cacheClearing: implementation.cacheClearing
    )
  }

  private static func load(
    detection: ModelDetection,
    engine: EngineKind,
    hardwareUnit: MLXHardwareUnit
  ) async throws -> Self {
    let directory = detection.directory
    switch (detection.model, engine) {
    case (.needle, .mlx):
      return try await Self.needleMLX(from: directory, hardwareUnit: hardwareUnit)
    case (.needle, .onnx): return try await Self.needleONNX(from: directory)
    case (.needle, .coreml): return try await Self.needleCoreML(from: directory)
    case (.needle, .coreai): return try await Self.needleCoreAI(from: directory)
    case (.qwen3, .mlx):
      return try await Self.llm(Qwen3MLXModel.self, from: directory, hardwareUnit: hardwareUnit)
    case (.qwen3P5, .mlx):
      return try await Self.llm(Qwen3P5MLXModel.self, from: directory, hardwareUnit: hardwareUnit)
    case (.lfm2, .mlx):
      return try await Self.llm(LFM2MLXModel.self, from: directory, hardwareUnit: hardwareUnit)
    case (.functionGemma, .mlx):
      return try await Self.llm(
        FunctionGemmaMLXModel.self,
        from: directory,
        hardwareUnit: hardwareUnit
      )
    case (.granite, .mlx):
      return try await Self.llm(GraniteMLXModel.self, from: directory, hardwareUnit: hardwareUnit)
    case (.graniteMoeHybrid, .mlx):
      return try await Self.llm(
        GraniteMoeHybridMLXModel.self,
        from: directory,
        hardwareUnit: hardwareUnit
      )
    case (.miniCPM5, .mlx):
      return try await Self.llm(MiniCPM5MLXModel.self, from: directory, hardwareUnit: hardwareUnit)
    default:
      throw EdgeCLIError("\(detection.model.displayName) does not support the \(engine.rawValue) engine.")
    }
  }

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
      supportsCustomGrammar: false,
      supportsSampling: true,
      usesMLX: true,
      prompt: needlePrompt,
      parameters: { request in
        NeedleMLXGenerateParameters(
          sampler: mlxSampler(for: request),
          maxTokens: request.maxTokens,
          toolCallRange: request.toolCallRange
        )
      },
      cacheClearing: { await engine.clearCaches() }
    )
  }

  private static func needleONNX(from directory: URL) async throws -> Self {
    let engine = try await NeedleCONNXModelEngine(from: directory)
    return Self(
      engineKind: .onnx,
      engine: engine,
      supportsCustomGrammar: false,
      supportsSampling: false,
      prompt: needlePrompt,
      parameters: { request in
        NeedleONNXGenerateParameters(
          maxTokens: request.maxTokens,
          toolCallRange: request.toolCallRange
        )
      },
      cacheClearing: { await engine.clearCaches() }
    )
  }

  private static func needleCoreML(from directory: URL) async throws -> Self {
    let engine = try await NeedleCoreMLModelEngine(
      modelDirectoryURL: directory,
      modelConfiguration: MLModelConfiguration()
    )
    return Self(
      engineKind: .coreml,
      engine: engine,
      supportsCustomGrammar: false,
      supportsSampling: false,
      prompt: needlePrompt,
      parameters: { request in
        NeedleCoreMLModel.GenerateParameters(
          maxTokens: request.maxTokens,
          toolCallRange: request.toolCallRange
        )
      },
      cacheClearing: { await engine.clearCaches() }
    )
  }

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
        supportsCustomGrammar: false,
        supportsSampling: false,
        prompt: needlePrompt,
        parameters: { request in
          NeedleCoreAIModel.GenerateParameters(
            maxTokens: request.maxTokens,
            toolCallRange: request.toolCallRange
          )
        },
        cacheClearing: { await engine.clearCaches() }
      )
    #else
      throw EdgeCLIError("This build of edge has no coreai engine; it requires Swift 6.4 or newer.")
    #endif
  }

  private static func llm<Model: MLXModel>(
    _ model: Model.Type,
    from directory: URL,
    hardwareUnit: MLXHardwareUnit
  ) async throws -> Self
  where
    Model.Prompt == EdgeToolsLLMPrompt,
    Model.GenerateParameters == DefaultMLXGenerateParameters,
    Model.GrammarCompiler == XGRCompiler,
    Model.GrammarContext == XGRGrammarContext,
    Model.ModelConfiguration: Decodable
  {
    let engine = try await Device.withDefaultDevice(hardwareUnit.device) {
      try await MLXEngine<Model>(from: directory)
    }
    return Self(
      engineKind: .mlx,
      engine: engine,
      supportsCustomGrammar: true,
      supportsSampling: true,
      usesMLX: true,
      prompt: { request in
        var messages = [EdgeToolsLLMPrompt.Message]()
        if !request.system.isEmpty { messages.append(.system(request.system)) }
        messages.append(.user(request.user))
        return EdgeToolsLLMPrompt(messages: messages)
      },
      parameters: { request in
        DefaultMLXGenerateParameters(
          sampler: mlxSampler(for: request),
          constraint: try request.grammar.constraint(toolCallRange: request.toolCallRange),
          maxTokens: request.maxTokens
        )
      },
      cacheClearing: { await engine.clearCaches() }
    )
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

private func mlxSampler(for request: GenerationRequest) -> any LogitSampler {
  guard request.temperature > 0 else { return ArgMaxSampler() }
  return TopPSampler(temperature: request.temperature, topP: request.topP)
}
