#if MLX && Transformers && canImport(MLX)
  import Foundation
  import MLXLLM

  extension MLXLLM.Gemma3TextModel: EdgeToolsLanguageModel {
    public typealias ModelConfiguration = MLXLLM.Gemma3TextConfiguration
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = FunctionGemmaToolCallParser

    public func grammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try XGRGrammar.functionGemma(tools: tools, range: range)
    }
  }

  public typealias FunctionGemmaMLXEngine = EdgeToolsMLXEngine<MLXLLM.Gemma3TextModel>

  extension EdgeToolsMLXEngine where Model == MLXLLM.Gemma3TextModel {
    public convenience init(from directoryURL: URL) async throws {
      try await self.init(from: directoryURL, model: MLXLLM.Gemma3TextModel.init)
    }
  }
#endif
