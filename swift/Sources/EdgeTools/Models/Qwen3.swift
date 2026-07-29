#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && Transformers && canImport(MLX)
  import _EdgeToolsFoundation
  import MLX
  import MLXLLM
  import MLXLMCommon

  // MARK: - Qwen3 Model

  extension Qwen3Model: EdgeToolsMLXModel, EdgeToolsPrefillableModel {
    public typealias ModelConfiguration = Qwen3Configuration
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias ToolCallParser = Qwen3ToolCallParser

    public func grammar(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      try .qwen3(tools: tools, range: range)
    }
  }

  /// The shared model engine specialized for Qwen 3.
  public typealias Qwen3MLXModelEngine = EdgeToolsModelEngine<Qwen3Model>

  // MARK: - Model Engine Loading

  extension EdgeToolsModelEngine where Model == Qwen3Model {
    /// Loads a Qwen 3 model engine from a model directory.
    public init(from directoryURL: URL) async throws {
      try await self.init(from: directoryURL, model: Qwen3Model.init)
    }
  }
#endif

// MARK: - Qwen3 Tool Call Parsing

public struct QwenJSONToolCallParser: EdgeToolCallParser, Sendable {
  private var block = IncrementalToolCallBlock(
    opener: "<tool_call>",
    closer: "</tool_call>"
  )

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> EdgeRawToolCall? {
    self.block.append(token)
    while let payload = self.block.nextPayload(respectingJSONStringBoundaries: true) {
      if let value = try? EdgeToolsJSONDecoder().decode(EdgeToolsValue.self, from: payload),
        let call = EdgeRawToolCall(jsonValue: value)
      {
        return call
      }
    }
    return nil
  }
}

public typealias Qwen3ToolCallParser = QwenJSONToolCallParser

// MARK: - Qwen3 Grammar

#if XGrammar
  extension XGRGrammar {
    public static func qwenJSON(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.toolCalls(tools: Array(tools), separator: "", range: range) {
        try Self.qwenJSONCall($0)
      }
    }

    public static func qwen3(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.qwenJSON(tools: tools, range: range)
    }

    private static func qwenJSONCall(_ tool: EdgeToolDefinition) throws -> XGRGrammar {
      let arguments = Self.strictJSONArguments(for: tool)
      let encodedName = OrderedKeyJSONWriter.encode(.string(tool.name))
      let prefix = try XGRGrammar.literal("<tool_call>{\"name\":\(encodedName),\"arguments\":")
      let withArguments = try prefix.concatenate(arguments)
      return try withArguments.concatenate(XGRGrammar.literal("}</tool_call>"))
    }
  }
#endif
