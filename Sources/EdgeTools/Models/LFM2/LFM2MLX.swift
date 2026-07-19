#if MLX && Transformers && canImport(MLX)
  #if canImport(FoundationEssentials)
    import FoundationEssentials
  #else
    import Foundation
  #endif
  import MLXLLM

  extension LFM2Model: EdgeToolsLanguageModel {
    public typealias ModelConfiguration = LFM2Configuration
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = LFM2ToolCallParser

    public func grammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try XGRGrammar.lfm2(tools: tools, range: range)
    }
  }

  public typealias LFM2MLXEngine = EdgeToolsMLXEngine<LFM2Model>
  public typealias LFM2P5MLXEngine = EdgeToolsMLXEngine<LFM2Model>

  extension EdgeToolsMLXEngine where Model == LFM2Model {
    public convenience init(from directoryURL: URL) async throws {
      try await self.init(from: directoryURL, model: LFM2Model.init)
    }
  }
#endif
