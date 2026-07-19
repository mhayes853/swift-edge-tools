#if MLX && Transformers && canImport(MLX)
  import Foundation
  import MLXLLM

  extension Qwen3Model: EdgeToolsLanguageModel {
    public typealias ModelConfiguration = Qwen3Configuration
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = Qwen3ToolCallParser

    public func grammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try XGRGrammar.qwen3(tools: tools, range: range)
    }
  }

  public typealias Qwen3MLXEngine = EdgeToolsMLXEngine<Qwen3Model>

  extension EdgeToolsMLXEngine where Model == Qwen3Model {
    public convenience init(from directoryURL: URL) async throws {
      try await self.init(from: directoryURL, model: Qwen3Model.init)
    }
  }
#endif
