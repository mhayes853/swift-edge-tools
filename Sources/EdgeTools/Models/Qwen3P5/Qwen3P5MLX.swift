#if MLX && Transformers && canImport(MLX)
  #if canImport(FoundationEssentials)
    import FoundationEssentials
  #else
    import Foundation
  #endif
  import MLXLLM

  extension Qwen35Model: EdgeToolsLanguageModel {
    public typealias ModelConfiguration = Qwen35Configuration
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = Qwen3P5ToolCallParser

    public func grammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try XGRGrammar.qwen3P5(tools: tools, range: range)
    }
  }

  public typealias Qwen35MLXEngine = EdgeToolsMLXEngine<Qwen35Model>
  public typealias Qwen3P5MLXEngine = EdgeToolsMLXEngine<Qwen35Model>

  extension EdgeToolsMLXEngine where Model == Qwen35Model {
    public convenience init(from directoryURL: URL) async throws {
      try await self.init(from: directoryURL, model: Qwen35Model.init)
    }
  }
#endif
