import EdgeToolsCore

/// Controls when a generation stream publishes accumulated text parts.
public struct EdgeToolsTextEmission: Sendable {
  /// The input available when deciding whether to publish pending text.
  public struct Pending: Sendable {
    /// Text accumulated since the previous text emission.
    public let payload: String
    /// The most recent token since the previous emission, when the engine publishes tokens.
    public let latestToken: EdgeToolsToken?
    /// Token events received since the previous text emission.
    public let tokenCount: Int
    /// Raw token and text-part events retained in the current buffer.
    public let events: [EdgeToolsGenerationStream.Event]
  }

  private let predicate: @Sendable (Pending) -> Bool

  /// Creates a policy that publishes the whole pending payload when the predicate succeeds.
  public init(shouldEmit: @escaping @Sendable (Pending) -> Bool) {
    self.predicate = shouldEmit
  }

  /// Publishes each text part as it arrives.
  public static let everyToken = Self { _ in true }

  /// Publishes after the specified number of token events. A count of one is equivalent to
  /// `everyToken`. For larger counts, engines without token events flush on completion.
  public static func every(tokenCount: Int) -> Self {
    precondition(tokenCount > 0, "The token count must be positive.")
    if tokenCount == 1 {
      return .everyToken
    }
    return Self { $0.tokenCount >= tokenCount }
  }

  /// Publishes the whole pending payload when it contains a newline.
  public static let onNewline = Self { $0.payload.contains("\n") }

  /// Publishes the whole pending payload when it contains a blank line.
  public static let onParagraphBreak = Self { $0.payload.contains("\n\n") }

  func shouldEmit(_ pending: Pending) -> Bool {
    self.predicate(pending)
  }
}
