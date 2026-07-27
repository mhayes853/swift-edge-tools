public enum GrammarToolCallRange: Hashable, Sendable {
  case unbounded(minimum: Int)
  case bounded(ClosedRange<Int>)
  case exact(Int)

  public static func unbounded(_ range: PartialRangeFrom<Int>) -> Self {
    .unbounded(minimum: range.lowerBound)
  }

  public static func bounded(_ range: PartialRangeThrough<Int>) -> Self {
    .bounded(0...range.upperBound)
  }

  public static func bounded(_ range: PartialRangeUpTo<Int>) -> Self {
    Self.bounded(lowerBound: 0, exclusiveUpperBound: range.upperBound)
  }

  public static func bounded(_ range: Range<Int>) -> Self {
    Self.bounded(lowerBound: range.lowerBound, exclusiveUpperBound: range.upperBound)
  }

  private static func bounded(lowerBound: Int, exclusiveUpperBound: Int) -> Self {
    if lowerBound < exclusiveUpperBound {
      .bounded(lowerBound...(exclusiveUpperBound - 1))
    } else {
      .exact(Swift.min(lowerBound, exclusiveUpperBound))
    }
  }

  public var lowerBound: Int {
    switch self {
    case .bounded(let range): range.lowerBound
    case .exact(let count): count
    case .unbounded(let minimum): minimum
    }
  }

  public var upperBound: Int? {
    switch self {
    case .bounded(let range): range.upperBound
    case .exact(let count): count
    case .unbounded: nil
    }
  }
}
