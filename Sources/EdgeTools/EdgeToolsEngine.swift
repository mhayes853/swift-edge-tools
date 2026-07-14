// MARK: - EdgeToolsEngine

public protocol EdgeToolsEngine: Sendable {
  associatedtype Prompt: EdgeToolsPrompt
  associatedtype GenerateParameters: EdgeToolsEngineGenerateParameters
  associatedtype GenerationTask: EdgeToolsEngineGenerationTask

  func tokenize(prompt: Prompt) async throws -> [EdgeToolsToken]

  func generate(
    prompt: Prompt,
    parameters: GenerateParameters,
    channel: EdgeToolsGenerationChannel
  ) throws -> GenerationTask
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
  public var metadata: EdgeToolsMetadata

  public init(
    prefillMetrics: EdgeToolsPrefillMetrics,
    decodeMetrics: EdgeToolsDecodeMetrics,
    wasStopped: Bool,
    tokens: [EdgeToolsToken],
    metadata: EdgeToolsMetadata = [:]
  ) {
    self.prefillMetrics = prefillMetrics
    self.decodeMetrics = decodeMetrics
    self.wasStopped = wasStopped
    self.tokens = tokens
    self.metadata = metadata
  }
}

extension EdgeToolsEngineGeneration {
  public static let empty = Self(
    prefillMetrics: EdgeToolsPrefillMetrics(tokens: 0, duration: .zero),
    decodeMetrics: EdgeToolsDecodeMetrics(tokens: 0, duration: .zero, durationToFirstToken: .zero),
    wasStopped: true,
    tokens: []
  )

  public var isEmpty: Bool {
    self.prefillMetrics == Self.empty.prefillMetrics
      && self.decodeMetrics == Self.empty.decodeMetrics
      && self.wasStopped
      && self.tokens.isEmpty
      && self.metadata.isEmpty
  }
}
