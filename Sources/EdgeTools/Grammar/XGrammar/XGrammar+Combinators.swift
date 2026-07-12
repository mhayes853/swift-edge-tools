#if XGrammar
  import CXGrammar

  public func concatenate(
    _ lhs: borrowing XGrammarGrammar,
    _ rhs: borrowing XGrammarGrammar
  ) throws -> XGrammarGrammar {
    let handles: [xgrammar_grammar_t?] = [lhs.handle, rhs.handle]
    let handle = try handles.withUnsafeBufferPointer {
      try XGrammarCompiler.requiredHandle(
        xgrammar_grammar_concatenate($0.baseAddress, $0.count)
      )
    }
    return XGrammarGrammar(handle: handle)
  }

  public func union(
    _ lhs: borrowing XGrammarGrammar,
    _ rhs: borrowing XGrammarGrammar
  ) throws -> XGrammarGrammar {
    let handles: [xgrammar_grammar_t?] = [lhs.handle, rhs.handle]
    let handle = try handles.withUnsafeBufferPointer {
      try XGrammarCompiler.requiredHandle(
        xgrammar_grammar_union($0.baseAddress, $0.count)
      )
    }
    return XGrammarGrammar(handle: handle)
  }

  public func `repeat`(
    _ grammar: borrowing XGrammarGrammar,
    exactly count: Int
  ) throws -> XGrammarGrammar {
    try `repeat`(grammar, count...count)
  }

  public func `repeat`(
    _ grammar: borrowing XGrammarGrammar,
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
    _ grammar: borrowing XGrammarGrammar,
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
