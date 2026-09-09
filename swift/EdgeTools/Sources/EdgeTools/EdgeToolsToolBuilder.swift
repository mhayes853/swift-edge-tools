// MARK: - EdgeToolsToolBuilder

@resultBuilder
public enum EdgeToolsToolBuilder {
  public static func buildExpression<Tool: EdgeTool>(_ tool: Tool) -> any EdgeTool {
    tool
  }

  public static func buildBlock(_ tools: any EdgeTool...) -> [any EdgeTool] {
    tools
  }

  public static func buildOptional(_ tools: [any EdgeTool]?) -> [any EdgeTool] {
    tools ?? []
  }

  public static func buildEither(first tools: [any EdgeTool]) -> [any EdgeTool] {
    tools
  }

  public static func buildEither(second tools: [any EdgeTool]) -> [any EdgeTool] {
    tools
  }

  public static func buildArray(_ tools: [[any EdgeTool]]) -> [any EdgeTool] {
    tools.flatMap { $0 }
  }
}
