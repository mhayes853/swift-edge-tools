import CoreML
import EdgeTools
import Foundation
import MLX
import MLXLMCommon

// MARK: - EngineRunner

public struct EngineRunner: Sendable {
  public let supportsCustomGrammar: Bool
  public let supportsSampling: Bool
  public let usesMLX: Bool

  private let generation:
    @Sendable (GenerationRequest, sending EdgeToolsGenerationChannel) async throws ->
      EdgeToolsEngineGeneration
  private let cacheClearing: @Sendable () async -> Void

  public init(
    supportsCustomGrammar: Bool,
    supportsSampling: Bool,
    usesMLX: Bool = false,
    generation:
      @escaping @Sendable (
        GenerationRequest, sending EdgeToolsGenerationChannel
      ) async throws -> EdgeToolsEngineGeneration,
    cacheClearing: @escaping @Sendable () async -> Void = {}
  ) {
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
    engine: Engine,
    supportsCustomGrammar: Bool,
    supportsSampling: Bool,
    usesMLX: Bool = false,
    prompt: @escaping @Sendable (GenerationRequest) -> Engine.Prompt,
    parameters: @escaping @Sendable (GenerationRequest) throws -> sending Engine.GenerateParameters,
    cacheClearing: @escaping @Sendable () async -> Void
  ) {
    self.init(
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
    engine: EngineKind,
    hardwareUnit: MLXHardwareUnit = .gpu
  ) async throws {
    let directory = detection.directory
    switch (detection.model, engine) {
    case (.needle, .mlx):
      self = try await Self.needleMLX(from: directory, hardwareUnit: hardwareUnit)
    case (.needle, .onnx): self = try await Self.needleONNX(from: directory)
    case (.needle, .coreml): self = try await Self.needleCoreML(from: directory)
    case (.needle, .coreai): self = try await Self.needleCoreAI(from: directory)
    case (.qwen3, _):
      self = try await Self.llm(Qwen3MLXModel.self, from: directory, hardwareUnit: hardwareUnit)
    case (.qwen3P5, _):
      self = try await Self.llm(Qwen3P5MLXModel.self, from: directory, hardwareUnit: hardwareUnit)
    case (.lfm2, _):
      self = try await Self.llm(LFM2MLXModel.self, from: directory, hardwareUnit: hardwareUnit)
    case (.functionGemma, _):
      self = try await Self.llm(
        FunctionGemmaMLXModel.self,
        from: directory,
        hardwareUnit: hardwareUnit
      )
    case (.granite, _):
      self = try await Self.llm(GraniteMLXModel.self, from: directory, hardwareUnit: hardwareUnit)
    case (.graniteMoeHybrid, _):
      self = try await Self.llm(
        GraniteMoeHybridMLXModel.self,
        from: directory,
        hardwareUnit: hardwareUnit
      )
    case (.miniCPM5, _):
      self = try await Self.llm(MiniCPM5MLXModel.self, from: directory, hardwareUnit: hardwareUnit)
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

private func needlePrompt(for request: GenerationRequest) -> NeedlePrompt {
  NeedlePrompt(system: request.system, user: request.user)
}

private func mlxSampler(for request: GenerationRequest) -> any LogitSampler {
  guard request.temperature > 0 else { return ArgMaxSampler() }
  return TopPSampler(temperature: request.temperature, topP: request.topP)
}
