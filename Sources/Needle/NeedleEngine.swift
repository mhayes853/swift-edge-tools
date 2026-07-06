// MARK: - NeedleEngine

public protocol NeedleEngine: Sendable {
  associatedtype GenerateParameters: NeedleEngineGenerateParameters
  associatedtype GenerationTask: NeedleEngineGenerationTask

  func tokenize(prompt: NeedlePrompt) async throws -> [NeedleToken]

  func generate(
    prompt: NeedlePrompt,
    parameters: GenerateParameters,
    onToken: @escaping @Sendable (NeedleToken) -> Void
  ) throws -> GenerationTask
}

// MARK: - NeedleEngineGenerateParemeters

public protocol NeedleEngineGenerateParameters: Sendable {
  static var `default`: Self { get }
}

// MARK: - NeedleEngineGenerationTask

public protocol NeedleEngineGenerationTask: Sendable {
  var value: NeedleEngineGeneration { get async throws }

  func stop()
}

// MARK: - NeedleEngineGeneration

public struct NeedleEngineGeneration: Sendable {
  public var prefillMetrics: NeedlePrefillMetrics
  public var decodeMetrics: NeedleDecodeMetrics
  public var wasStopped: Bool
  public var tokens: [NeedleToken]
  public var metadata: NeedleMetadata

  public init(
    prefillMetrics: NeedlePrefillMetrics,
    decodeMetrics: NeedleDecodeMetrics,
    wasStopped: Bool,
    tokens: [NeedleToken],
    metadata: NeedleMetadata = [:]
  ) {
    self.prefillMetrics = prefillMetrics
    self.decodeMetrics = decodeMetrics
    self.wasStopped = wasStopped
    self.tokens = tokens
    self.metadata = metadata
  }
}

extension NeedleEngineGeneration {
  public static let empty = Self(
    prefillMetrics: NeedlePrefillMetrics(tokens: 0, duration: .zero),
    decodeMetrics: NeedleDecodeMetrics(tokens: 0, duration: .zero, durationToFirstToken: .zero),
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
