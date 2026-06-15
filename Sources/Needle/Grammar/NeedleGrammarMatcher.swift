public protocol NeedleGrammarMatcher {
  func bitmask() -> NeedleGrammarBitmask
  @discardableResult
  mutating func accept(tokenId: NeedleToken.ID) -> Bool
}
