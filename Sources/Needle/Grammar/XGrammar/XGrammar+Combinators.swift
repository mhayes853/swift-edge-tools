#if XGrammar
  import CXGrammar

  public func concatenate(
    _ grammars: XGrammarGrammar...
  ) throws -> XGrammarGrammar {
    try concatenate(contentsOf: grammars)
  }

  public func concatenate(
    contentsOf grammars: some Sequence<XGrammarGrammar>
  ) throws -> XGrammarGrammar {
    let grammars = Array(grammars)
    guard !grammars.isEmpty else { throw XGrammarError.emptyGrammarCollection }
    let handles: [xgrammar_grammar_t?] = grammars.map { $0.handle }
    let handle = try handles.withUnsafeBufferPointer {
      try XGrammarCompiler.requiredHandle(
        xgrammar_grammar_concatenate($0.baseAddress, $0.count)
      )
    }
    return XGrammarGrammar(handle: handle)
  }

  public func union(
    _ grammars: XGrammarGrammar...
  ) throws -> XGrammarGrammar {
    try union(contentsOf: grammars)
  }

  public func union(
    contentsOf grammars: some Sequence<XGrammarGrammar>
  ) throws -> XGrammarGrammar {
    let grammars = Array(grammars)
    guard !grammars.isEmpty else { throw XGrammarError.emptyGrammarCollection }
    let handles: [xgrammar_grammar_t?] = grammars.map { $0.handle }
    let handle = try handles.withUnsafeBufferPointer {
      try XGrammarCompiler.requiredHandle(
        xgrammar_grammar_union($0.baseAddress, $0.count)
      )
    }
    return XGrammarGrammar(handle: handle)
  }

  public func `repeat`(
    _ grammar: XGrammarGrammar,
    exactly count: Int
  ) throws -> XGrammarGrammar {
    try `repeat`(grammar, count...count)
  }

  public func `repeat`(
    _ grammar: XGrammarGrammar,
    _ range: ClosedRange<Int>
  ) throws -> XGrammarGrammar {
    guard range.lowerBound >= 0,
      Int32(exactly: range.lowerBound) != nil,
      Int32(exactly: range.upperBound) != nil
    else {
      throw XGrammarError.invalidRepetitionRange
    }
    let handle = try XGrammarCompiler.requiredHandle(
      xgrammar_grammar_repeat(
        grammar.handle,
        Int32(range.lowerBound),
        Int32(range.upperBound)
      )
    )
    return XGrammarGrammar(handle: handle)
  }

  public func `repeat`(
    _ grammar: XGrammarGrammar,
    _ range: PartialRangeFrom<Int>
  ) throws -> XGrammarGrammar {
    guard range.lowerBound >= 0, Int32(exactly: range.lowerBound) != nil else {
      throw XGrammarError.invalidRepetitionRange
    }
    let handle = try XGrammarCompiler.requiredHandle(
      xgrammar_grammar_repeat(grammar.handle, Int32(range.lowerBound), -1)
    )
    return XGrammarGrammar(handle: handle)
  }
#endif
