// MARK: - EdgeToolsPrompt

public protocol EdgeToolsPrompt: Sendable {
  var tools: [any EdgeTool] { get }
}
