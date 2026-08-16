import EdgeToolsCore

public struct EdgeToolsGenerationChannel {
  public var onToken: ((EdgeToolsToken) -> Void)?
  public var onPart: ((EdgeToolsGenerationPart) -> Void)?

  public init(
    onToken: ((EdgeToolsToken) -> Void)? = nil,
    onPart: ((EdgeToolsGenerationPart) -> Void)? = nil
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
