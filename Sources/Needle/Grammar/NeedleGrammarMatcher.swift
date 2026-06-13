public protocol NeedleGrammarMatcher {
  func bitmask() -> NeedleGrammarBitmask
  mutating func accept(tokenId: NeedleToken.ID)
}
