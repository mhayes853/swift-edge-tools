import CoreML
import EdgeTools
import Foundation
import MLXLMCommon

// MARK: - GenerationRequest

public struct GenerationRequest: Sendable {
  public var system: String
  public var user: String
  public var tools: [EdgeToolDefinition]
  public var grammar: GrammarOption
  public var toolCallRange: GrammarToolCallRange
  public var maxTokens: Int?
  public var temperature: Float
  public var topP: Float

  public init(
    system: String,
    user: String,
    tools: [EdgeToolDefinition],
    grammar: GrammarOption,
    toolCallRange: GrammarToolCallRange,
    maxTokens: Int?,
    temperature: Float,
    topP: Float
  ) {
    self.system = system
    self.user = user
    self.tools = tools
    self.grammar = grammar
    self.toolCallRange = toolCallRange
    self.maxTokens = maxTokens
    self.temperature = temperature
    self.topP = topP
  }
}

// MARK: - EdgeRunner

public protocol EdgeRunner: Sendable {
  /// Whether the underlying generate parameters expose a full generation constraint, rather than
  /// only a tool call range.
  var supportsCustomGrammar: Bool { get }

  /// Whether the underlying generate parameters expose sampling controls.
  var supportsSampling: Bool { get }

  func generate(
    _ request: GenerationRequest,
    channel: EdgeToolsGenerationChannel
  ) async throws -> EdgeToolsEngineGeneration

  /// Discards cached grammar state so the next generation starts from a comparable position.
  func reset() async
}

// MARK: - EngineRunner

struct EngineRunner<Engine: EdgeToolsEngine>: EdgeRunner {
  let engine: Engine
  let clearCaches: @Sendable () async -> Void
  let supportsCustomGrammar: Bool
  let supportsSampling: Bool
  let makePrompt: @Sendable (GenerationRequest) -> Engine.Prompt
  let makeParameters: @Sendable (GenerationRequest) throws -> sending Engine.GenerateParameters

  func reset() async {
    await self.clearCaches()
  }

  func generate(
    _ request: GenerationRequest,
    channel: EdgeToolsGenerationChannel
  ) async throws -> EdgeToolsEngineGeneration {
    let task = try self.engine.generate(
      prompt: self.makePrompt(request),
      tools: request.tools,
      parameters: try self.makeParameters(request),
      channel: channel
    )
    return try await task.value
  }
}

// MARK: - Loading

public func makeRunner(
  detection: ModelDetection,
  engine kind: EngineKind
) async throws -> any EdgeRunner {
  let directory = detection.directory
  switch (detection.model, kind) {
  case (.needle, .mlx):
    let engine = try await NeedleMLXModelEngine(from: directory)
    return EngineRunner(
      engine: engine,
      clearCaches: { await engine.clearCaches() },
      supportsCustomGrammar: false,
      supportsSampling: true,
      makePrompt: needlePrompt,
      makeParameters: { request in
        NeedleMLXGenerateParameters(
          sampler: mlxSampler(request),
          maxTokens: request.maxTokens,
          toolCallRange: request.toolCallRange
        )
      }
    )

  case (.needle, .onnx):
    let engine = try await NeedleCONNXModelEngine(from: directory)
    return EngineRunner(
      engine: engine,
      clearCaches: { await engine.clearCaches() },
      supportsCustomGrammar: false,
      supportsSampling: false,
      makePrompt: needlePrompt,
      makeParameters: { request in
        NeedleONNXGenerateParameters(
          maxTokens: request.maxTokens,
          toolCallRange: request.toolCallRange
        )
      }
    )

  case (.needle, .coreml):
    let engine = try await NeedleCoreMLModelEngine(
      modelDirectoryURL: directory,
      modelConfiguration: MLModelConfiguration()
    )
    return EngineRunner(
      engine: engine,
      clearCaches: { await engine.clearCaches() },
      supportsCustomGrammar: false,
      supportsSampling: false,
      makePrompt: needlePrompt,
      makeParameters: { request in
        NeedleCoreMLModel.GenerateParameters(
          maxTokens: request.maxTokens,
          toolCallRange: request.toolCallRange
        )
      }
    )

  case (.needle, .coreai):
    return try await makeNeedleCoreAIRunner(from: directory)

  case (.qwen3, _):
    return try await makeDefaultMLXRunner(from: directory, engine: Qwen3MLXModelEngine.init)

  case (.qwen35, _):
    return try await makeDefaultMLXRunner(from: directory, engine: Qwen35MLXModelEngine.init)

  case (.lfm2, _):
    return try await makeDefaultMLXRunner(from: directory, engine: LFM2MLXModelEngine.init)

  case (.functionGemma, _):
    return try await makeDefaultMLXRunner(
      from: directory,
      engine: FunctionGemmaMLXModelEngine.init
    )

  case (.genericLLM, _):
    return try await makeGenericLLMRunner(from: directory)
  }
}

// MARK: - Helpers

private func needlePrompt(_ request: GenerationRequest) -> NeedlePrompt {
  NeedlePrompt(system: request.system, user: request.user)
}

func llmPrompt(_ request: GenerationRequest) -> EdgeToolsLLMPrompt {
  var messages = [EdgeToolsLLMPrompt.Message]()
  if !request.system.isEmpty {
    messages.append(.system(request.system))
  }
  messages.append(.user(request.user))
  return EdgeToolsLLMPrompt(messages: messages)
}

func mlxSampler(_ request: GenerationRequest) -> any LogitSampler {
  guard request.temperature > 0 else { return ArgMaxSampler() }
  return TopPSampler(temperature: request.temperature, topP: request.topP)
}

func defaultMLXParameters(
  _ request: GenerationRequest
) throws -> sending DefaultEdgeToolsMLXGenerateParameters {
  DefaultEdgeToolsMLXGenerateParameters(
    sampler: mlxSampler(request),
    constraint: try request.grammar.constraint(toolCallRange: request.toolCallRange),
    maxTokens: request.maxTokens
  )
}

private func makeDefaultMLXRunner<Model: EdgeToolsMLXModel>(
  from directory: URL,
  engine makeEngine: @Sendable (URL) async throws -> EdgeToolsMLXEngine<Model>
) async throws -> any EdgeRunner
where
  Model.Prompt == EdgeToolsLLMPrompt,
  Model.GenerateParameters == DefaultEdgeToolsMLXGenerateParameters,
  Model.GrammarCompiler == XGRCompiler,
  Model.GrammarContext == XGRGrammarContext
{
  let engine = try await makeEngine(directory)
  return EngineRunner(
    engine: engine,
    clearCaches: { await engine.clearCaches() },
    supportsCustomGrammar: true,
    supportsSampling: true,
    makePrompt: llmPrompt,
    makeParameters: defaultMLXParameters
  )
}
