#if XGrammar
  import Foundation

  extension XGRGrammar {
    public static func lfm2(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.lfm2Python(tools: tools, range: range)
    }

    public static func lfm2P5(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try Self.lfm2Python(tools: tools, range: range)
    }

    public static func lfm2Python(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      let calls = try Self.toolCalls(tools: Array(tools), separator: ",", range: range) {
        try Self.pythonCall($0)
      }
      let prefix = try XGRGrammar(literal: "<|tool_call_start|>[")
      let withCalls = try prefix.concatenate(calls)
      let grammar = try withCalls.concatenate(
        XGRGrammar(literal: "]<|tool_call_end|>")
      )
      var document = try XGREBNFDocument(grammar.ebnf)
      document.removeDuplicateRules()
      return try XGRGrammar(ebnf: document.source)
    }

    private static func pythonCall(_ tool: EdgeToolDefinition) throws -> XGRGrammar {
      let jsonArguments = Self.strictJSONArguments(for: tool)
      var document = try XGREBNFDocument(jsonArguments.ebnf)
      try document.mapLiterals { ruleName, value, suffix in
        switch value {
        case "true": return "True"
        case "false": return "False"
        case "null": return "None"
        default: break
        }

        guard Self.isLFMTopLevelArgumentRule(ruleName) else { return value }
        if value == "{" || value == "}" { return "" }
        if value == ":" { return "=" }
        if value.count >= 2, value.first == "\"", value.last == "\"",
          suffix.contains(#"":""#)
        {
          return String(value.dropFirst().dropLast())
        }
        return value
      }

      let arguments = try XGRGrammar(ebnf: document.source)
      let prefix = try XGRGrammar(literal: "\(tool.name)(")
      let withArguments = try prefix.concatenate(arguments)
      return try withArguments.concatenate(XGRGrammar(literal: ")"))
    }

    private static func isLFMTopLevelArgumentRule(_ name: String) -> Bool {
      name == "root" || name.wholeMatch(of: lfmTopLevelArgumentRuleRegex) != nil
    }
  }

  nonisolated(unsafe) private let lfmTopLevelArgumentRuleRegex =
    /root_(?:[0-9]+|part_[0-9]+)/
#endif
