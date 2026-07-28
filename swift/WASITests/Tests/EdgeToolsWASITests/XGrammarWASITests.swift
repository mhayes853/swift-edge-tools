import EdgeTools
import Testing

@Suite
struct `XGrammar WASI tests` {
  @Test
  func `Creates Builtin JSON Grammar`() {
    let grammar = XGRGrammar.builtinJSONGrammar()

    #expect(grammar.ebnf.contains("root"))
  }

  @Test
  func `Parses Supported Grammar Formats`() throws {
    let ebnf = try XGRGrammar.ebnf("root ::= \"edge\"")
    let regex = try XGRGrammar.regex("edge(tools)?")
    let lark = try XGRGrammar.lark("start: \"edge\"")
    let jsonSchema = try XGRGrammar.jsonSchema(#"{"type":"object","properties":{"value":{"type":"integer"}}}"#)

    #expect(ebnf.ebnf.contains("edge"))
    #expect(regex.ebnf.contains("root"))
    #expect(lark.ebnf.contains("root"))
    #expect(jsonSchema.ebnf.contains("root"))
  }

  @Test
  func `Invalid Serialized Grammar Throws`() {
    #expect(throws: XGRError.self) {
      _ = try XGRGrammar.serializedJSON("not json")
    }
  }

  @Test
  func `Invalid Structural Tag Throws`() {
    #expect(throws: XGRError.self) {
      _ = try XGRGrammar.structuralTagJSON("not json")
    }
  }

  @Test
  func `Compiles And Matches Arithmetic Operators`() throws {
    let grammar = try XGRGrammar.lark(
      """
        %import common.WS_INLINE
        %import common.NUMBER
        %ignore WS_INLINE
        start: expression
        ?expression: term (("+" | "-") term)*
        ?term: factor (("*" | "/") factor)*
        ?factor: NUMBER | "-" factor | "(" expression ")"
        """
    )
    let compiler = try self.testCompiler()
    let compiledGrammar = try compiler.compile(grammar)
    let matcher = try XGRMatcher(
      compiledGrammar: compiledGrammar,
      terminateWithoutStopToken: true
    )

    let accepted = matcher.accept(string: "12 + 3 * (4 - 1)")
    let isTerminated = matcher.isTerminated

    #expect(compiledGrammar.memorySizeBytes > 0)
    #expect(accepted)
    #expect(isTerminated)
  }

  @Test
  func `Compiles Query Grammar With Logical And Comparison Operators`() throws {
    let grammar = try XGRGrammar.ebnf(
      #"""
        root ::= "SELECT" ws fields ws "FROM" ws identifier (ws "WHERE" ws predicate)?
        fields ::= "*" | identifier (ws "," ws identifier)*
        predicate ::= identifier ws comparison ws value (ws logical ws identifier ws comparison ws value)*
        comparison ::= "=" | "!=" | "<" | "<=" | ">" | ">="
        logical ::= "AND" | "OR"
        value ::= number | quoted
        identifier ::= [A-Za-z_][A-Za-z0-9_]*
        number ::= "-"? [0-9]+ ("." [0-9]+)?
        quoted ::= "\"" [A-Za-z0-9 ]* "\""
        ws ::= [ \t\n]*
        """#
    )
    let compiler = try self.testCompiler()
    let compiledGrammar = try compiler.compile(grammar)

    #expect(compiledGrammar.memorySizeBytes > 0)
  }

  @Test
  func `Compiles Nested Command JSON Schema`() throws {
    let grammar = try XGRGrammar.jsonSchema(
      #"""
        {
          "type": "object",
          "required": ["command", "payload"],
          "properties": {
            "command": { "enum": ["search", "calculate"] },
            "payload": {
              "anyOf": [
                {
                  "type": "object",
                  "required": ["query", "limit"],
                  "properties": {
                    "query": { "type": "string", "minLength": 1 },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 100 },
                    "filters": {
                      "type": "array",
                      "items": { "type": "string" },
                      "maxItems": 5
                    }
                  },
                  "additionalProperties": false
                },
                {
                  "type": "object",
                  "required": ["expression"],
                  "properties": {
                    "expression": { "type": "string", "pattern": "^[0-9+*/(). -]+$" },
                    "variables": {
                      "type": "object",
                      "additionalProperties": { "type": "number" }
                    }
                  },
                  "additionalProperties": false
                }
              ]
            }
          },
          "additionalProperties": false
        }
        """#
    )
    let compiler = try self.testCompiler()
    let compiledGrammar = try compiler.compile(grammar)

    #expect(compiledGrammar.memorySizeBytes > 0)
  }

  private func testCompiler() throws -> XGRCompiler {
    let vocabulary = (32...126).map { String(UnicodeScalar($0)!) } + [""]
    let tokenizerInfo = try XGRTokenizerInfo(
      encodedVocabulary: vocabulary,
      vocabularyType: .raw,
      stopTokenIDs: [vocabulary.count - 1]
    )
    return try XGRCompiler(tokenizerInfo: tokenizerInfo)
  }
}
