public enum NeedleGrammarToolCallRange: Hashable, Sendable {
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
    .bounded(0..<range.upperBound)
  }

  public static func bounded(_ range: Range<Int>) -> Self {
    .bounded(range.lowerBound...(range.upperBound - 1))
  }
}
