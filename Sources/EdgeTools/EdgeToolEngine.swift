// MARK: - EdgeToolEngine

public protocol EdgeToolEngine: Sendable {
  associatedtype GenerateParameters: EdgeToolEngineGenerateParameters
  associatedtype GenerationTask: EdgeToolEngineGenerationTask

  func tokenize(prompt: EdgeToolsPrompt) async throws -> [EdgeToolsToken]

  func generate(
    prompt: EdgeToolsPrompt,
    parameters: GenerateParameters,
    onToken: @escaping @Sendable (EdgeToolsToken, EdgeRawToolCall?) -> Void
  ) throws -> GenerationTask
}

// MARK: - EdgeToolEngineGenerateParemeters

public protocol EdgeToolEngineGenerateParameters: Sendable {
  static var `default`: Self { get }
}

// MARK: - EdgeToolEngineGenerationTask

public protocol EdgeToolEngineGenerationTask: Sendable {
  var value: EdgeToolEngineGeneration { get async throws }

  func stop()
}

// MARK: - EdgeToolEngineGeneration

public struct EdgeToolEngineGeneration: Sendable {
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

extension EdgeToolEngineGeneration {
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
