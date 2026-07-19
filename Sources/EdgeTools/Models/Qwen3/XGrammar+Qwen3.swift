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
      let encodedName = String(decoding: encodeEdgeToolsJSONString(tool.name), as: UTF8.self)
      let prefix = try XGRGrammar(
        literal: "<tool_call>{\"name\":\(encodedName),\"arguments\":"
      )
      let withArguments = try prefix.concatenate(arguments)
      return try withArguments.concatenate(XGRGrammar(literal: "}</tool_call>"))
    }
  }
#endif
