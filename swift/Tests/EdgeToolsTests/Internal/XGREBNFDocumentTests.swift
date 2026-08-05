#if XGrammar
  import CustomDump
  import Testing

  @testable import EdgeTools

  // NB: XGrammar does not currently emit EBNF that reaches these cases, so nothing here is
  // observable through the grammar fixtures. They exist because the scanner is hand-rolled and
  // telling literals apart from rule references is the invariant a future simplification would
  // quietly drop.
  @Suite
  struct `XGREBNFDocument tests` {
    @Test
    func `Maps Literals That Are Directly Adjacent To Each Other`() throws {
      var document = try XGREBNFDocument(#"root ::= "{""}""#)
      var visited = [String]()
      try document.mapLiterals { _, value, _ in
        visited.append(value)
        return value == "{" ? "[" : "]"
      }
      expectNoDifference(visited, ["{", "}"])
      expectNoDifference(document.source, #"root ::= "[""]""#)
    }

    @Test
    func `Keeps Rule References Inside Adjacent Literals Intact While Deduplicating`() throws {
      var document = try XGREBNFDocument(
        """
        root ::= alpha beta
        alpha ::= "x"
        beta ::= "x"
        gamma ::= "alpha""beta"
        """
      )
      document.removeDuplicateRules()
      let body = { name in document.rules.first { $0.name == name }?.body }
      expectNoDifference(body("root"), "alpha alpha")
      expectNoDifference(body("gamma"), #""alpha""beta""#)
      expectNoDifference(body("beta"), nil)
    }

    @Test
    func `Ignores Continuation References That Only Appear Inside Literals`() throws {
      expectNoDifference(try firstLiteralSuffixHasContinuation(#"root ::= "a" "root""#), false)
    }

    @Test
    func `Detects Continuation References Outside Of Literals`() throws {
      expectNoDifference(try firstLiteralSuffixHasContinuation(#"root ::= "a" root_1"#), true)
    }
  }

  // MARK: - Helpers

  private func firstLiteralSuffixHasContinuation(_ source: String) throws -> Bool? {
    var document = try XGREBNFDocument(source)
    var firstSuffix: Bool?
    try document.mapLiterals { _, value, suffix in
      if firstSuffix == nil { firstSuffix = suffix.hasToolCallContinuationReference }
      return value
    }
    return firstSuffix
  }
#endif
