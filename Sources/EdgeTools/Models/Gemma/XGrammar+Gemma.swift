#if XGrammar
  extension XGrammarGrammar {
    public static func functionGemma(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGrammarGrammar {
      try Self.toolCalls(tools: Array(tools), separator: "", range: range) {
        try Self.functionGemmaCall($0)
      }
    }

    public static func gemma4(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGrammarGrammar {
      try Self.toolCalls(tools: Array(tools), separator: "", range: range) {
        try Self.gemma4Call($0)
      }
    }

    private static func functionGemmaCall(
      _ tool: EdgeToolDefinition
    ) throws -> XGrammarGrammar {
      let xmlArguments = Self.qwenXMLArguments(for: tool)
      var document = try XGrammarEBNFDocument(xmlArguments.ebnf)
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

      let arguments = try XGrammarGrammar(ebnf: document.source)
      let prefix = try XGrammarGrammar(
        literal: "<start_function_call>call:\(tool.name){"
      )
      let withArguments = try prefix.concatenate(arguments)
      return try withArguments.concatenate(
        XGrammarGrammar(literal: "}<end_function_call>")
      )
    }

    private static func gemma4Call(_ tool: EdgeToolDefinition) throws -> XGrammarGrammar {
      let jsonArguments = Self.strictJSONArguments(for: tool)
      var document = try XGrammarEBNFDocument(jsonArguments.ebnf)
      try document.mapLiterals { _, value, suffix in
        let isBeforePropertyColon = suffix.contains(#"":""#)
        if value == "\"" {
          return isBeforePropertyColon ? "" : "<|\"|>"
        }
        if value.count >= 2, value.first == "\"", value.last == "\"" {
          let inner = value.dropFirst().dropLast()
          return isBeforePropertyColon ? String(inner) : "<|\"|>\(inner)<|\"|>"
        }
        return value
      }
      guard let stringIndex = document.rules.firstIndex(where: { $0.name == "basic_string" })
      else { throw ToolCallXGrammarError.unsupportedSchema }
      document.rules[stringIndex].body =
        #"(("<|\"|>" gemma_string_content "<|\"|>"))"#
      if let objectIndex = document.rules.firstIndex(where: { $0.name == "basic_object" }),
        let keyRange = document.rules[objectIndex].body.range(of: "basic_string")
      {
        document.rules[objectIndex].body.replaceSubrange(keyRange, with: "gemma_key")
      }
      document.rules.append(
        XGrammarEBNFDocument.Rule(
          name: "gemma_string_content",
          body: #"TagDispatch(loop_after_dispatch=false, excludes=("<|\"|>"))"#
        )
      )
      document.rules.append(
        XGrammarEBNFDocument.Rule(
          name: "gemma_key",
          body: #"TagDispatch(loop_after_dispatch=false, excludes=(":"))"#
        )
      )

      let arguments = try XGrammarGrammar(ebnf: document.source)
      let prefix = try XGrammarGrammar(literal: "<|tool_call>call:\(tool.name)")
      let withArguments = try prefix.concatenate(arguments)
      return try withArguments.concatenate(XGrammarGrammar(literal: "<tool_call|>"))
    }
  }
#endif
