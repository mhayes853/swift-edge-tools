import EdgeToolsCore

// MARK: - EdgeToolsEngineContext

public protocol EdgeToolsEngineContext: Identifiable, Sendable {
  var tools: [any EdgeTool] { get }
}

// MARK: - EdgeToolsEngine

public protocol EdgeToolsEngine: Sendable {
  associatedtype Context: EdgeToolsEngineContext where Context.ID: Sendable
  associatedtype ContextParameters: Sendable = Void
  associatedtype Prompt: Sendable
  associatedtype GenerateParameters: EdgeToolsEngineGenerateParameters
  associatedtype GenerationTask: EdgeToolsEngineGenerationTask

  func context(tools: [any EdgeTool]) -> Context
  func context(_ parameters: ContextParameters, tools: [any EdgeTool]) -> Context

  func generate(
    prompt: Prompt,
    parameters: sending GenerateParameters,
    context: Context,
    channel: sending EdgeToolsGenerationChannel
  ) throws -> GenerationTask
}

extension EdgeToolsEngine {
  public func context() -> Context {
    self.context(tools: [])
  }
}

extension EdgeToolsEngine where ContextParameters == Void {
  public func context(tools: [any EdgeTool]) -> Context {
    self.context((), tools: tools)
  }
}

extension EdgeToolsEngine {
  public func context<Parameters>() -> Context where ContextParameters == Parameters? {
    self.context(nil, tools: [])
  }

  public func context(_ parameters: ContextParameters) -> Context {
    self.context(parameters, tools: [])
  }

  public func context(
    @EdgeToolsToolBuilder tools: () -> [any EdgeTool]
  ) -> Context {
    self.context(tools: tools())
  }

  public func context(
    _ parameters: ContextParameters,
    @EdgeToolsToolBuilder tools: () -> [any EdgeTool]
  ) -> Context {
    self.context(parameters, tools: tools())
  }
}

// MARK: - EdgeToolsTokenizingEngine

public protocol EdgeToolsTokenizingEngine: EdgeToolsEngine {
  func tokenize(prompt: Prompt, context: Context) async throws -> [EdgeToolsToken]
}

extension EdgeToolsTokenizingEngine {
  public func tokenize(prompt: Prompt) async throws -> [EdgeToolsToken] {
    try await self.tokenize(prompt: prompt, context: self.context())
  }
}

// MARK: - EdgeToolsPrefillableEngine

public protocol EdgeToolsPrefillableEngine: EdgeToolsEngine {
  func prefill(promptPrefix: Prompt, context: Context) async throws -> EdgeToolsEnginePrefill
}

// MARK: - EdgeToolsEnginePrefill

public struct EdgeToolsEnginePrefill: Sendable {
  public var metrics: EdgeToolsMetrics

  public init(metrics: EdgeToolsMetrics) {
    self.metrics = metrics
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
  public var wasStopped: Bool
  public var tokens: [EdgeToolsToken]
  public var response: String
  public var parts: [EdgeToolsGenerationPart]
  public var metrics: EdgeToolsMetrics

  public init(
    wasStopped: Bool,
    tokens: [EdgeToolsToken],
    response: String,
    parts: [EdgeToolsGenerationPart] = [],
    metrics: EdgeToolsMetrics = [:]
  ) {
    self.wasStopped = wasStopped
    self.tokens = tokens
    self.response = response
    self.parts = parts
    self.metrics = metrics
  }
}

extension EdgeToolsEngineGeneration {
  public static let empty = Self(wasStopped: true, tokens: [], response: "")

  public var isEmpty: Bool {
    self.wasStopped
      && self.tokens.isEmpty
      && self.response.isEmpty
      && self.parts.isEmpty
      && self.metrics.isEmpty
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
