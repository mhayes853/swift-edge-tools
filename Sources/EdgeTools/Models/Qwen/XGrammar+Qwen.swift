#if XGrammar
  import Foundation

  extension XGrammarGrammar {
    public static func qwenJSON(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGrammarGrammar {
      try Self.toolCalls(tools: Array(tools), separator: "", range: range) {
        try Self.qwenJSONCall($0)
      }
    }

    public static func qwenXML(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGrammarGrammar {
      try Self.toolCalls(tools: Array(tools), separator: "", range: range) {
        try Self.qwenXMLCall($0)
      }
    }

    public static func qwen3(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGrammarGrammar {
      try Self.qwenJSON(tools: tools, range: range)
    }

    public static func qwen3P5(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGrammarGrammar {
      try Self.qwenXML(tools: tools, range: range)
    }

    public static func qwen3P6(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGrammarGrammar {
      try Self.qwenXML(tools: tools, range: range)
    }

    private static func qwenJSONCall(_ tool: EdgeToolDefinition) throws -> XGrammarGrammar {
      let arguments = try Self.strictJSONArguments(for: tool)
      let encodedName = String(decoding: try JSONEncoder().encode(tool.name), as: UTF8.self)
      let prefix = try XGrammarGrammar(
        literal: "<tool_call>{\"name\":\(encodedName),\"arguments\":"
      )
      let withArguments = try prefix.concatenate(arguments)
      return try withArguments.concatenate(XGrammarGrammar(literal: "}</tool_call>"))
    }

    private static func qwenXMLCall(_ tool: EdgeToolDefinition) throws -> XGrammarGrammar {
      let arguments = try Self.qwenXMLArguments(for: tool)
      let prefix = try XGrammarGrammar(literal: "<tool_call><function=\(tool.name)>")
      let withArguments = try prefix.concatenate(arguments)
      return try withArguments.concatenate(
        XGrammarGrammar(literal: "</function></tool_call>")
      )
    }
  }
#endif
