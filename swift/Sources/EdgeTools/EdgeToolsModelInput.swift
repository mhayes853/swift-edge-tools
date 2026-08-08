// MARK: - EdgeToolsModelInput

public protocol EdgeToolsModelInput {
  var tokenIds: [EdgeToolsToken.ID] { get }
}

extension Array: EdgeToolsModelInput where Element == EdgeToolsToken.ID {
  public var tokenIds: [EdgeToolsToken.ID] { self }
}
