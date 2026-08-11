// MARK: - EdgeToolsEngine

public protocol EdgeToolsEngine: Sendable {
  associatedtype Context: Identifiable & Sendable where Context.ID: Sendable
  associatedtype ContextParameters: Sendable = Void
  associatedtype Prompt: Sendable
  associatedtype GenerateParameters: EdgeToolsEngineGenerateParameters
  associatedtype GenerationTask: EdgeToolsEngineGenerationTask

  func context() -> Context
  func context(_ parameters: ContextParameters) -> Context

  func tokenize(
    prompt: Prompt,
    tools: [EdgeToolDefinition],
    context: Context
  ) async throws -> [EdgeToolsToken]

  func generate(
    prompt: Prompt,
    tools: [EdgeToolDefinition],
    parameters: sending GenerateParameters,
    context: Context,
    channel: sending EdgeToolsGenerationChannel
  ) throws -> GenerationTask
}

extension EdgeToolsEngine where ContextParameters == Void {
  public func context() -> Context {
    self.context(())
  }
}

extension EdgeToolsEngine {
  public func context<Parameters>() -> Context
  where ContextParameters == Parameters? {
    self.context(nil)
  }

  public func tokenize(
    prompt: Prompt,
    tools: [EdgeToolDefinition] = []
  ) async throws -> [EdgeToolsToken] {
    try await self.tokenize(
      prompt: prompt,
      tools: tools,
      context: self.context()
    )
  }

  public func generate(
    prompt: Prompt,
    tools: [EdgeToolDefinition] = [],
    parameters: sending GenerateParameters,
    channel: sending EdgeToolsGenerationChannel
  ) throws -> GenerationTask {
    try self.generate(
      prompt: prompt,
      tools: tools,
      parameters: parameters,
      context: self.context(),
      channel: channel
    )
  }
}

// MARK: - EdgeToolsPrefillableEngine

public protocol EdgeToolsPrefillableEngine: EdgeToolsEngine {
  func prefill(
    promptPrefix: Prompt,
    tools: [EdgeToolDefinition],
    context: Context
  ) async throws -> EdgeToolsEnginePrefill
}

extension EdgeToolsPrefillableEngine {
  public func prefill(
    promptPrefix: Prompt,
    tools: [EdgeToolDefinition]
  ) async throws -> EdgeToolsEnginePrefill {
    try await self.prefill(
      promptPrefix: promptPrefix,
      tools: tools,
      context: self.context()
    )
  }
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

public protocol EdgeToolsEngineGenerateParameters {
  static var `default`: Self { get }

  var maxTokens: Int? { get }
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
  public var parts: [EdgeToolsGenerationPart]
  public var metadata: EdgeToolsMetadata

  public init(
    prefillMetrics: EdgeToolsPrefillMetrics,
    decodeMetrics: EdgeToolsDecodeMetrics,
    wasStopped: Bool,
    tokens: [EdgeToolsToken],
    response: String,
    parts: [EdgeToolsGenerationPart] = [],
    metadata: EdgeToolsMetadata = [:]
  ) {
    self.prefillMetrics = prefillMetrics
    self.decodeMetrics = decodeMetrics
    self.wasStopped = wasStopped
    self.tokens = tokens
    self.response = response
    self.parts = parts
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
      && self.parts.isEmpty
      && self.metadata.isEmpty
  }

  public var text: String {
    guard !self.parts.isEmpty else { return self.response }
    return self.parts.compactMap { $0.text }.joined()
  }

  public var reasoning: [String] {
    self.parts.compactMap { $0.reasoning }
  }

  public var toolCalls: [EdgeRawToolCall] {
    self.parts.compactMap { $0.toolCall }
  }
}
