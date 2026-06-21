// MARK: - NeedleEngine

public protocol NeedleEngine {
  associatedtype GrammarEngine: NeedleGrammarEngine
  associatedtype GenerateParameters: NeedleEngineGenerateParameters

  var grammarEngine: GrammarEngine { get }

  var stopper: NeedleEngineStopper { get }

  func generate(
    prompt: NeedlePrompt,
    matcher: GrammarEngine.Matcher,
    parameters: GenerateParameters,
    onToken: (NeedleToken) -> Void
  ) throws -> NeedleEngineGeneration

  func reset()
}

// MARK: - NeedleEngineGenerateParemeters

public protocol NeedleEngineGenerateParameters {
  static var `default`: Self { get }
}

// MARK: - NeedleEngineStopper

public struct NeedleEngineStopper: Sendable {
  private let stop: @Sendable () -> Void

  public init(stop: @escaping @Sendable () -> Void) {
    self.stop = stop
  }

  public func callAsFunction() {
    self.stop()
  }
}

// MARK: - NeedleEngineGeneration

public struct NeedleEngineGeneration: Sendable {
  public var prefillMetrics: NeedlePrefillMetrics
  public var decodeMetrics: NeedleDecodeMetrics
  public var wasStoped: Bool
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
    self.wasStoped = wasStopped
    self.tokens = tokens
    self.metadata = metadata
  }
}
