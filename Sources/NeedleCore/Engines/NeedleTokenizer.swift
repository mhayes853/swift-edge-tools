public struct NeedleTokenizer: Sendable {

  public func encode(text: String) -> [NeedleToken] {
    []
  }

  public func encodeToTokenIds(text: String) -> [Int] {
    self.encode(text: text).map(\.id)
  }

  public func decode(tokenIds: some Sequence<Int>) -> [NeedleToken] {
    []
  }

  public func decodeToString(tokenIds: some Sequence<Int>) -> String {
    self.decode(tokenIds: tokenIds).reduce(into: "") { $0 += $1.stringValue }
  }
}
