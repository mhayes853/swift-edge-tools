import EdgeToolsCore

public protocol EdgeToolsGenerationParser {
  init()

  mutating func accept(token: EdgeToolsToken) -> [EdgeToolsGenerationPart]
  mutating func finish() -> [EdgeToolsGenerationPart]
}
