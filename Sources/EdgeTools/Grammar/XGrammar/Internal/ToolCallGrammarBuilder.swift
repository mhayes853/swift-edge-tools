#if XGrammar
  extension XGRGrammar {
    static func strictJSONArguments(for tool: EdgeToolDefinition) -> XGRGrammar {
      guard
        let grammar = try? XGRGrammar(
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

    static func qwenXMLArguments(for tool: EdgeToolDefinition) -> XGRGrammar {
      let schema = tool.arguments.orderedKeyEncoded()
      let structuralTag =
        #"{"type":"structural_tag","format":{"type":"json_schema","json_schema":\#(schema),"style":"qwen_xml","any_order":false}}"#
      guard let grammar = try? XGRGrammar(structuralTagJSON: structuralTag) else {
        preconditionFailure("Edge tool arguments must produce a valid Qwen XML structural tag.")
      }
      return grammar
    }

    static func toolCalls(
      tools: [EdgeToolDefinition],
      separator: String,
      range: GrammarToolCallRange,
      call: (EdgeToolDefinition) throws -> XGRGrammar
    ) throws -> XGRGrammar {
      guard range.lowerBound >= 0 else { throw ToolCallXGRError.invalidToolInvocationRange }
      guard let firstTool = tools.first else {
        guard range.lowerBound == 0 else { throw ToolCallXGRError.emptyToolCollection }
        return try XGRGrammar(literal: "")
      }

      var grammar = try call(firstTool)
      for tool in tools.dropFirst() {
        grammar = try grammar.union(call(tool))
      }
      return try Self.repeatingToolCall(grammar, separator: separator, range: range)
    }

    static func repeatingToolCall(
      _ call: borrowing XGRGrammar,
      separator: String,
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      guard range.lowerBound >= 0 else { throw ToolCallXGRError.invalidToolInvocationRange }
      let separatedCall = try XGRGrammar(literal: separator).concatenate(call)

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
        guard minimum >= 0 else { throw ToolCallXGRError.invalidToolInvocationRange }
        let tail = try separatedCall.repeated(Swift.max(0, minimum - 1)...)
        let nonempty = try call.concatenate(tail)
        return minimum == 0 ? try nonempty.optional() : nonempty
      }
    }

    private static func boundedToolCalls(
      _ call: borrowing XGRGrammar,
      separatedCall: borrowing XGRGrammar,
      minimum: Int,
      maximum: Int
    ) throws -> XGRGrammar {
      guard minimum >= 0, maximum >= minimum else {
        throw ToolCallXGRError.invalidToolInvocationRange
      }
      guard maximum > 0 else { return try XGRGrammar(literal: "") }
      let tail = try separatedCall.repeated(Swift.max(0, minimum - 1)...(maximum - 1))
      let nonempty = try call.concatenate(tail)
      return minimum == 0 ? try nonempty.optional() : nonempty
    }
  }

  public enum ToolCallXGRError: Error, Hashable, Sendable {
    case invalidToolInvocationRange
    case emptyToolCollection
    case unsupportedSchema
  }
#endif
