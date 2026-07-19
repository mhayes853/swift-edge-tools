#if XGrammar
  extension XGRGrammar {
    public static func functionGemma(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.toolCalls(tools: Array(tools), separator: "", range: range) {
        try Self.functionGemmaCall($0)
      }
    }

    private static func functionGemmaCall(
      _ tool: EdgeToolDefinition
    ) throws -> XGRGrammar {
      let xmlArguments = Self.qwenXMLArguments(for: tool)
      var document = try XGREBNFDocument(xmlArguments.ebnf)
      try document.mapLiterals { ruleName, value, suffix in
        if value == "</parameter>" {
          if ruleName == "xml_string" { return "<escape>" }
          return suffix.hasToolCallContinuationReference ? "<escape>," : "<escape>"
        }
        if value.hasPrefix("<parameter="), value.hasSuffix(">") {
          let name = value.dropFirst("<parameter=".count).dropLast()
          return "\(name):<escape>"
        }
        if value == "<parameter=" { return "" }
        if value == ">", ruleName.hasPrefix("xml_object") {
          return ":<escape>"
        }
        return value
      }

      let arguments = try XGRGrammar(ebnf: document.source)
      let prefix = try XGRGrammar(
        literal: "<start_function_call>call:\(tool.name){"
      )
      let withArguments = try prefix.concatenate(arguments)
      return try withArguments.concatenate(
        XGRGrammar(literal: "}<end_function_call>")
      )
    }

  }
#endif
