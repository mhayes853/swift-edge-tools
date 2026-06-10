// MARK: - NeedleStaticToolsCollection

public protocol NeedleStaticToolsCollection<Output> {
  associatedtype Output: NeedleStaticToolsCollectionOutput

  static var tools: [any NeedleTool] { get }
}

// MARK: - NeedleStaticToolsCollectionOutput

public protocol NeedleStaticToolsCollectionOutput {
  func call<Tool: NeedleTool>(for tool: Tool.Type) -> NeedleToolCallOf<Tool>?
}
