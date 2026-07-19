#if XGrammar
  extension XGRGrammar {
    public static func qwenXML(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.toolCalls(tools: Array(tools), separator: "", range: range) {
        try Self.qwenXMLCall($0)
      }
    }

    public static func qwen3P5(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.qwenXML(tools: tools, range: range)
    }

    public static func qwen3P6(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.qwenXML(tools: tools, range: range)
    }

    private static func qwenXMLCall(_ tool: EdgeToolDefinition) throws -> XGRGrammar {
      let arguments = Self.qwenXMLArguments(for: tool)
      let prefix = try XGRGrammar(literal: "<tool_call><function=\(tool.name)>")
      let withArguments = try prefix.concatenate(arguments)
      return try withArguments.concatenate(
        XGRGrammar(literal: "</function></tool_call>")
      )
    }
  }
#endif
