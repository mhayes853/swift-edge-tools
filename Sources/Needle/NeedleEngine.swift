// MARK: - NeedleEngine

public protocol NeedleEngine {
  associatedtype GrammarEngine: NeedleGrammarEngine

  var grammarEngine: GrammarEngine { get }

  var stopper: NeedleEngineStopper { get }

  func generate(
    prompt: NeedlePrompt,
    matcher: GrammarEngine.Matcher,
    onToken: (NeedleToken) -> Void
  ) throws -> NeedleEngineGeneration

  func reset()
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

public struct NeedleEngineGeneration: Hashable, Sendable {
  public let prefillMetrics: NeedlePrefillMetrics
  public let decodeMetrics: NeedleDecodeMetrics
  public let wasStoped: Bool
  public let response: String

  public init(
    prefillMetrics: NeedlePrefillMetrics,
    decodeMetrics: NeedleDecodeMetrics,
    wasStopped: Bool,
    response: String
  ) {
    self.prefillMetrics = prefillMetrics
    self.decodeMetrics = decodeMetrics
    self.wasStoped = wasStopped
    self.response = response
  }
}
