#if XGrammar
  import CXGrammar

  public func concatenate(
    _ lhs: borrowing XGRGrammar,
    _ rhs: borrowing XGRGrammar
  ) throws -> XGRGrammar {
    let handles: [xgrammar_grammar_t?] = [lhs.handle, rhs.handle]
    let handle = try handles.withUnsafeBufferPointer {
      try xgrammarRequiredHandle(xgrammar_grammar_concatenate($0.baseAddress, $0.count))
    }
    return XGRGrammar(handle: handle)
  }

  public func union(
    _ lhs: borrowing XGRGrammar,
    _ rhs: borrowing XGRGrammar
  ) throws -> XGRGrammar {
    let handles: [xgrammar_grammar_t?] = [lhs.handle, rhs.handle]
    let handle = try handles.withUnsafeBufferPointer {
      try xgrammarRequiredHandle(xgrammar_grammar_union($0.baseAddress, $0.count))
    }
    return XGRGrammar(handle: handle)
  }

  public func repeatGrammar(
    _ grammar: borrowing XGRGrammar,
    exactly count: Int
  ) throws -> XGRGrammar {
    try repeatGrammar(grammar, count...count)
  }

  public func repeatGrammar(
    _ grammar: borrowing XGRGrammar,
    _ range: ClosedRange<Int>
  ) throws -> XGRGrammar {
    guard range.lowerBound >= 0 else { throw XGRError.invalidRepetitionRange }
    let lowerBound = try xgrammarInt32(range.lowerBound, error: .invalidRepetitionRange)
    let upperBound = try xgrammarInt32(range.upperBound, error: .invalidRepetitionRange)
    let handle = try xgrammarRequiredHandle(
      xgrammar_grammar_repeat(grammar.handle, lowerBound, upperBound)
    )
    return XGRGrammar(handle: handle)
  }

  public func repeatGrammar(
    _ grammar: borrowing XGRGrammar,
    _ range: PartialRangeFrom<Int>
  ) throws -> XGRGrammar {
    guard range.lowerBound >= 0 else { throw XGRError.invalidRepetitionRange }
    let lowerBound = try xgrammarInt32(range.lowerBound, error: .invalidRepetitionRange)
    let handle = try xgrammarRequiredHandle(
      xgrammar_grammar_repeat(grammar.handle, lowerBound, -1)
    )
    return XGRGrammar(handle: handle)
  }
#endif
