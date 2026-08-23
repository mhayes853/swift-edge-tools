extension UInt8 {
  package static let jsonOpenArray = UInt8(ascii: "[")
  package static let jsonCloseArray = UInt8(ascii: "]")
  package static let jsonOpenObject = UInt8(ascii: "{")
  package static let jsonCloseObject = UInt8(ascii: "}")
  package static let jsonComma = UInt8(ascii: ",")
  package static let jsonColon = UInt8(ascii: ":")
  package static let jsonQuote = UInt8(ascii: "\"")
  package static let jsonEscape = UInt8(ascii: "\\")

  package var isASCIIWhitespace: Bool {
    self == UInt8(ascii: " ") || self == UInt8(ascii: "\n")
      || self == UInt8(ascii: "\r") || self == UInt8(ascii: "\t")
  }
}
