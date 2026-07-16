#if XGrammar
  import CXGrammar

  public func concatenate(
    _ lhs: borrowing XGrammarGrammar,
    _ rhs: borrowing XGrammarGrammar
  ) throws -> XGrammarGrammar {
    let handles: [xgrammar_grammar_t?] = [lhs.handle, rhs.handle]
    let handle = try handles.withUnsafeBufferPointer {
      try xgrammarRequiredHandle(xgrammar_grammar_concatenate($0.baseAddress, $0.count))
    }
    return XGrammarGrammar(handle: handle)
  }

  public func union(
    _ lhs: borrowing XGrammarGrammar,
    _ rhs: borrowing XGrammarGrammar
  ) throws -> XGrammarGrammar {
    let handles: [xgrammar_grammar_t?] = [lhs.handle, rhs.handle]
    let handle = try handles.withUnsafeBufferPointer {
      try xgrammarRequiredHandle(xgrammar_grammar_union($0.baseAddress, $0.count))
    }
    return XGrammarGrammar(handle: handle)
  }

  public func repeatGrammar(
    _ grammar: borrowing XGrammarGrammar,
    exactly count: Int
  ) throws -> XGrammarGrammar {
    try repeatGrammar(grammar, count...count)
  }

  public func repeatGrammar(
    _ grammar: borrowing XGrammarGrammar,
    _ range: ClosedRange<Int>
  ) throws -> XGrammarGrammar {
    guard range.lowerBound >= 0 else { throw XGrammarError.invalidRepetitionRange }
    let lowerBound = try xgrammarInt32(range.lowerBound, error: .invalidRepetitionRange)
    let upperBound = try xgrammarInt32(range.upperBound, error: .invalidRepetitionRange)
    let handle = try xgrammarRequiredHandle(
      xgrammar_grammar_repeat(grammar.handle, lowerBound, upperBound)
    )
    return XGrammarGrammar(handle: handle)
  }

  public func repeatGrammar(
    _ grammar: borrowing XGrammarGrammar,
    _ range: PartialRangeFrom<Int>
  ) throws -> XGrammarGrammar {
    guard range.lowerBound >= 0 else { throw XGrammarError.invalidRepetitionRange }
    let lowerBound = try xgrammarInt32(range.lowerBound, error: .invalidRepetitionRange)
    let handle = try xgrammarRequiredHandle(
      xgrammar_grammar_repeat(grammar.handle, lowerBound, -1)
    )
    return XGrammarGrammar(handle: handle)
  }
#endif
