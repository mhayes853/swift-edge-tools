#if XGrammar
  import Foundation

  #if canImport(Tokenizers)
    import Tokenizers

    extension XGrammarCompiler {
      public typealias Matcher = XGrammarMatcher

      public static func needle(
        tokenizer: any Tokenizer,
        configuration: Configuration = Configuration()
      ) -> XGrammarCompiler? {
        guard let eosTokenID = tokenizer.eosTokenId else { return nil }
        do {
          return try XGrammarCompiler(
            encodedVocabulary: tokenizer.convertIdsToTokens(Array(0..<8192)).compactMap { $0 },
            vocabularyType: .byteFallback,
            vocabularySize: 8192,
            stopTokenIDs: [eosTokenID],
            addPrefixSpace: true,
            configuration: configuration
          )
        } catch {
          return nil
        }
      }
    }
  #endif

  extension XGrammarGrammar {
    public static func needle(
      tools: some Sequence<NeedleToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGrammarGrammar {
      let tools = tools.map { $0.normalized() }
      let calls = try XGrammarGrammar.needleCalls(tools: tools, range: range)
      return try Needle.concatenate(contentsOf: [
        XGrammarGrammar(literal: "<tool_call> ["),
        calls,
        XGrammarGrammar(literal: "]")
      ])
    }

    private static func needleCalls(
      tools: [NeedleToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGrammarGrammar {
      guard range.lowerBound >= 0 else { throw NeedleXGrammarError.invalidToolInvocationRange }
      guard !tools.isEmpty else { return try XGrammarGrammar(literal: "") }

      let calls = try tools.map(XGrammarGrammar.needleCall)
      let call = try Needle.union(contentsOf: calls)
      let separatedCall = try Needle.concatenate(contentsOf: [XGrammarGrammar(literal: ","), call])

      switch range {
      case .exact(let count):
        return try XGrammarGrammar.needleCallList(
          call: call,
          separatedCall: separatedCall,
          minimum: count,
          maximum: count
        )
      case .bounded(let bounds):
        return try XGrammarGrammar.needleCallList(
          call: call,
          separatedCall: separatedCall,
          minimum: bounds.lowerBound,
          maximum: bounds.upperBound
        )
      case .unbounded(let minimum):
        return try XGrammarGrammar.needleUnboundedCallList(
          call: call,
          separatedCall: separatedCall,
          minimum: minimum
        )
      }
    }

    private static func needleCall(_ tool: NeedleToolDefinition) throws -> XGrammarGrammar {
      let schema = String(decoding: tool.arguments.needleGrammarEncoded(), as: UTF8.self)
      let arguments = try XGrammarGrammar(
        jsonSchema: schema,
        configuration: JSONSchemaConfiguration(
          anyWhitespace: false,
          separators: .init(comma: ",", colon: ":"),
          isStrict: true
        )
      )
      return try Needle.concatenate(contentsOf: [
        XGrammarGrammar(literal: "{\"name\":\""),
        XGrammarGrammar(literal: tool.name),
        XGrammarGrammar(literal: "\",\"arguments\":"),
        arguments,
        XGrammarGrammar(literal: "}")
      ])
    }

    private static func needleCallList(
      call: XGrammarGrammar,
      separatedCall: XGrammarGrammar,
      minimum: Int,
      maximum: Int
    ) throws -> XGrammarGrammar {
      guard minimum >= 0, maximum >= minimum else {
        throw NeedleXGrammarError.invalidToolInvocationRange
      }
      guard maximum > 0 else { return try XGrammarGrammar(literal: "") }
      let repeatedCalls = try separatedCall.repeated((Swift.max(0, minimum - 1))...(maximum - 1))
      let nonempty = try Needle.concatenate(contentsOf: [call, repeatedCalls])
      return minimum == 0 ? try nonempty.optional() : nonempty
    }

    private static func needleUnboundedCallList(
      call: XGrammarGrammar,
      separatedCall: XGrammarGrammar,
      minimum: Int
    ) throws -> XGrammarGrammar {
      guard minimum >= 0 else { throw NeedleXGrammarError.invalidToolInvocationRange }
      let repeatedCalls = try separatedCall.repeated(Swift.max(0, minimum - 1)...)
      let nonempty = try Needle.concatenate(contentsOf: [call, repeatedCalls])
      return minimum == 0 ? try nonempty.optional() : nonempty
    }
  }

  public enum NeedleXGrammarError: Error, Hashable, Sendable {
    case invalidToolInvocationRange
  }
#endif
