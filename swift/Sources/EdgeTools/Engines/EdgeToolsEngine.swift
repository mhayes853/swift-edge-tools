// MARK: - EdgeToolsEngine

public protocol EdgeToolsEngine: Sendable {
  associatedtype Prompt: Sendable
  associatedtype GenerateParameters: EdgeToolsEngineGenerateParameters
  associatedtype GenerationTask: EdgeToolsEngineGenerationTask

  func tokenize(prompt: Prompt, tools: [EdgeToolDefinition]) async throws -> [EdgeToolsToken]

  func generate(
    prompt: Prompt,
    tools: [EdgeToolDefinition],
    parameters: GenerateParameters,
    channel: EdgeToolsGenerationChannel
  ) throws -> GenerationTask
}

// MARK: - EdgeToolsPrefillableEngine

public protocol EdgeToolsPrefillableEngine: EdgeToolsEngine {
  func prefill(
    promptPrefix: Prompt,
    tools: [EdgeToolDefinition]
  ) async throws -> EdgeToolsEnginePrefill
}

// MARK: - EdgeToolsEnginePrefill

public struct EdgeToolsEnginePrefill: Sendable {
  public var metrics: EdgeToolsPrefillMetrics
  public var metadata: EdgeToolsMetadata

  public init(
    metrics: EdgeToolsPrefillMetrics,
    metadata: EdgeToolsMetadata = [:]
  ) {
    self.metrics = metrics
    self.metadata = metadata
  }
}

// MARK: - EdgeToolsEngineGenerateParemeters

public protocol EdgeToolsEngineGenerateParameters: Sendable {
  static var `default`: Self { get }
}

// MARK: - EdgeToolsEngineGenerationTask

public protocol EdgeToolsEngineGenerationTask: Sendable {
  var value: EdgeToolsEngineGeneration { get async throws }

  func stop()
}

// MARK: - EdgeToolsEngineGeneration

public struct EdgeToolsEngineGeneration: Sendable {
  public var prefillMetrics: EdgeToolsPrefillMetrics
  public var decodeMetrics: EdgeToolsDecodeMetrics
  public var wasStopped: Bool
  public var tokens: [EdgeToolsToken]
  public var response: String
  public var metadata: EdgeToolsMetadata

  public init(
    prefillMetrics: EdgeToolsPrefillMetrics,
    decodeMetrics: EdgeToolsDecodeMetrics,
    wasStopped: Bool,
    tokens: [EdgeToolsToken],
    response: String,
    metadata: EdgeToolsMetadata = [:]
  ) {
    self.prefillMetrics = prefillMetrics
    self.decodeMetrics = decodeMetrics
    self.wasStopped = wasStopped
    self.tokens = tokens
    self.response = response
    self.metadata = metadata
  }
}

extension EdgeToolsEngineGeneration {
  public static let empty = Self(
    prefillMetrics: EdgeToolsPrefillMetrics(tokens: 0, duration: .zero),
    decodeMetrics: EdgeToolsDecodeMetrics(tokens: 0, duration: .zero, durationToFirstToken: .zero),
    wasStopped: true,
    tokens: [],
    response: ""
  )

  public var isEmpty: Bool {
    self.prefillMetrics == Self.empty.prefillMetrics
      && self.decodeMetrics == Self.empty.decodeMetrics
      && self.wasStopped
      && self.tokens.isEmpty
      && self.response.isEmpty
      && self.metadata.isEmpty
  }
}
