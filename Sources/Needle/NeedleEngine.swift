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
  public var response: String
  public var metadata: [MetadataKey: any Sendable]

  public init(
    prefillMetrics: NeedlePrefillMetrics,
    decodeMetrics: NeedleDecodeMetrics,
    wasStopped: Bool,
    response: String,
    metadata: [MetadataKey: any Sendable]
  ) {
    self.prefillMetrics = prefillMetrics
    self.decodeMetrics = decodeMetrics
    self.wasStoped = wasStopped
    self.response = response
    self.metadata = metadata
  }
}

extension NeedleEngineGeneration {
  public struct MetadataKey: Hashable, Sendable, RawRepresentable, ExpressibleByStringLiteral {
    public var rawValue: String

    public init(rawValue: String) {
      self.rawValue = rawValue
    }

    public init(stringLiteral value: StringLiteralType) {
      self.init(rawValue: value)
    }
  }
}
