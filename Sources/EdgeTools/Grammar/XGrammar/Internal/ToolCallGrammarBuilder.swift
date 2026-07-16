#if XGrammar
  extension XGrammarGrammar {
    static func strictJSONArguments(for tool: EdgeToolDefinition) -> XGrammarGrammar {
      guard
        let grammar = try? XGrammarGrammar(
          jsonSchema: tool.arguments.orderedKeyEncoded(),
          configuration: JSONSchemaConfiguration(
            anyWhitespace: false,
            separators: .init(comma: ",", colon: ":"),
            isStrict: true
          )
        )
      else {
        preconditionFailure("Edge tool arguments must produce a valid JSON Schema.")
      }
      return grammar
    }

    static func qwenXMLArguments(for tool: EdgeToolDefinition) -> XGrammarGrammar {
      let schema = tool.arguments.orderedKeyEncoded()
      let structuralTag =
        #"{"type":"structural_tag","format":{"type":"json_schema","json_schema":\#(schema),"style":"qwen_xml","any_order":false}}"#
      guard let grammar = try? XGrammarGrammar(structuralTagJSON: structuralTag) else {
        preconditionFailure("Edge tool arguments must produce a valid Qwen XML structural tag.")
      }
      return grammar
    }

    static func toolCalls(
      tools: [EdgeToolDefinition],
      separator: String,
      range: GrammarToolCallRange,
      call: (EdgeToolDefinition) throws -> XGrammarGrammar
    ) throws -> XGrammarGrammar {
      guard range.lowerBound >= 0 else { throw ToolCallXGrammarError.invalidToolInvocationRange }
      guard let firstTool = tools.first else {
        guard range.lowerBound == 0 else { throw ToolCallXGrammarError.emptyToolCollection }
        return try XGrammarGrammar(literal: "")
      }

      var grammar = try call(firstTool)
      for tool in tools.dropFirst() {
        grammar = try grammar.union(call(tool))
      }
      return try Self.repeatingToolCall(grammar, separator: separator, range: range)
    }

    static func repeatingToolCall(
      _ call: borrowing XGrammarGrammar,
      separator: String,
      range: GrammarToolCallRange
    ) throws -> XGrammarGrammar {
      guard range.lowerBound >= 0 else { throw ToolCallXGrammarError.invalidToolInvocationRange }
      let separatedCall = try XGrammarGrammar(literal: separator).concatenate(call)

      switch range {
      case .exact(let count):
        return try Self.boundedToolCalls(
          call,
          separatedCall: separatedCall,
          minimum: count,
          maximum: count
        )
      case .bounded(let bounds):
        return try Self.boundedToolCalls(
          call,
          separatedCall: separatedCall,
          minimum: bounds.lowerBound,
          maximum: bounds.upperBound
        )
      case .unbounded(let minimum):
        guard minimum >= 0 else { throw ToolCallXGrammarError.invalidToolInvocationRange }
        let tail = try separatedCall.repeated(Swift.max(0, minimum - 1)...)
        let nonempty = try call.concatenate(tail)
        return minimum == 0 ? try nonempty.optional() : nonempty
      }
    }

    private static func boundedToolCalls(
      _ call: borrowing XGrammarGrammar,
      separatedCall: borrowing XGrammarGrammar,
      minimum: Int,
      maximum: Int
    ) throws -> XGrammarGrammar {
      guard minimum >= 0, maximum >= minimum else {
        throw ToolCallXGrammarError.invalidToolInvocationRange
      }
      guard maximum > 0 else { return try XGrammarGrammar(literal: "") }
      let tail = try separatedCall.repeated(Swift.max(0, minimum - 1)...(maximum - 1))
      let nonempty = try call.concatenate(tail)
      return minimum == 0 ? try nonempty.optional() : nonempty
    }
  }

  public enum ToolCallXGrammarError: Error, Hashable, Sendable {
    case invalidToolInvocationRange
    case emptyToolCollection
    case unsupportedSchema
  }
#endif
