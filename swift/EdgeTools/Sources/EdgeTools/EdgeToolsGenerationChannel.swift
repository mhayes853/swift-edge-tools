import EdgeToolsCore

public struct EdgeToolsGenerationChannel: Sendable {
  public var onToken: (@Sendable (EdgeToolsToken) -> Void)?
  public var onPart: (@Sendable (EdgeToolsGenerationPart) -> Void)?

  public init(
    onToken: (@Sendable (EdgeToolsToken) -> Void)? = nil,
    onPart: (@Sendable (EdgeToolsGenerationPart) -> Void)? = nil
  ) {
    self.onToken = onToken
    self.onPart = onPart
  }

  public func emit(token: EdgeToolsToken) {
    self.onToken?(token)
  }

  public func emit(part: EdgeToolsGenerationPart) {
    self.onPart?(part)
  }
}
