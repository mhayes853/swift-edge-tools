public protocol NeedleGrammarMatcher {
  func bitmask() -> NeedleGrammarBitmask
  func accept(token: NeedleToken)
}
