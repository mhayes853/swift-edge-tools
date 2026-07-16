#if XGrammar && Sentencepiece
  import CustomDump
  import EdgeTools
  import Testing

  @Suite(.serialized)
  struct `LFM XGrammarCompiler tests`: ~Copyable {
    private let compiler: XGrammarCompiler
    private let tokenizer: EdgeToolsSPTokenizer
    private let eosToken: EdgeToolsToken.ID

    init() throws {
      let tokenizer = try makeTestTokenizer()
      let compiler = try makeGenericXGrammarCompiler(tokenizer: tokenizer)
      let eosToken = try requiredTestEOSToken(tokenizer: tokenizer)
      self.tokenizer = tokenizer
      self.compiler = compiler
      self.eosToken = eosToken
    }

    @Test
    func `LFM Python Removes Duplicate EBNF Rules`() throws {
      let grammar = try XGrammarGrammar.lfmPython(
        tools: [.getWeather, Self.getForecast],
        range: .exact(1)
      )
      let expressions = try Self.expressions(in: grammar.ebnf)

      expectNoDifference(Set(expressions).count, expressions.count)
    }

    @Test
    func `LFM Python Uses Keyword Arguments And Nested Dictionaries`() throws {
      let validMatcher = try self.compiler.compile(
        try XGrammarGrammar.lfmPython(tools: [Self.nestedTool], range: .exact(1))
      )
      let valid =
        """
        <|tool_call_start|>[nested(payload={"enabled":True,"child":{"missing":None,"items":[\
        {"flag":False}]}})]<|tool_call_end|>
        """
      assertGrammarAccepts(
        valid,
        matcher: validMatcher,
        tokenizer: self.tokenizer,
        eosToken: self.eosToken
      )

      let optionalMatcher = try self.compiler.compile(
        try XGrammarGrammar.lfmPython(tools: [Self.nestedTool], range: .exact(1))
      )
      assertGrammarAccepts(
        """
        <|tool_call_start|>[nested(payload={"enabled":True,"child":{"missing":None,"items":[]}},\
        note="hello")]<|tool_call_end|>
        """,
        matcher: optionalMatcher,
        tokenizer: self.tokenizer,
        eosToken: self.eosToken
      )

      let topLevelJSONMatcher = try self.compiler.compile(
        try XGrammarGrammar.lfmPython(tools: [Self.nestedTool], range: .exact(1))
      )
      assertGrammarRejects(
        #"<|tool_call_start|>[nested("payload":{"enabled":True,"child":{"missing":None,"items":[]}})]<|tool_call_end|>"#,
        matcher: topLevelJSONMatcher,
        tokenizer: self.tokenizer,
        eosToken: self.eosToken
      )

      let nestedKeywordMatcher = try self.compiler.compile(
        try XGrammarGrammar.lfmPython(tools: [Self.nestedTool], range: .exact(1))
      )
      assertGrammarRejects(
        #"<|tool_call_start|>[nested(payload={"enabled"=True,"child":{"missing":None,"items":[]}})]<|tool_call_end|>"#,
        matcher: nestedKeywordMatcher,
        tokenizer: self.tokenizer,
        eosToken: self.eosToken
      )
    }

    private static func expressions(in ebnf: String) -> [String] {
      ebnf.split(separator: "\n").compactMap { line in
        guard let separator = line.range(of: "::=") else { return nil }
        return line[separator.upperBound...].trimmingCharacters(in: .whitespaces)
      }
    }

    private static var getForecast: EdgeToolDefinition {
      var tool = EdgeToolDefinition.getWeather
      tool.name = "getForecast"
      return tool
    }

    private static let nestedTool = EdgeToolDefinition(
      name: "nested",
      description: "Exercises nested Python values.",
      arguments: EdgeToolsGenerationSchema(
        .type(.object),
        .properties([
          "payload": EdgeToolsGenerationSchema(
            .type(.object),
            .properties([
              "enabled": .boolean,
              "child": EdgeToolsGenerationSchema(
                .type(.object),
                .properties([
                  "missing": .null,
                  "items": EdgeToolsGenerationSchema(
                    .type(.array),
                    .items(
                      EdgeToolsGenerationSchema(
                        .type(.object),
                        .properties(["flag": .boolean]),
                        .required(["flag"]),
                        .additionalProperties(false)
                      )
                    )
                  )
                ]),
                .required(["missing", "items"]),
                .additionalProperties(false)
              )
            ]),
            .required(["enabled", "child"]),
            .additionalProperties(false)
          ),
          "note": .string
        ]),
        .required(["payload"]),
        .additionalProperties(false)
      )
    )
  }
#endif
