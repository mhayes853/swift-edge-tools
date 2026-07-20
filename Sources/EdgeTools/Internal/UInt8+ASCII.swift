extension UInt8 {
  static let jsonOpenArray = UInt8(ascii: "[")
  static let jsonCloseArray = UInt8(ascii: "]")
  static let jsonOpenObject = UInt8(ascii: "{")
  static let jsonCloseObject = UInt8(ascii: "}")
  static let jsonComma = UInt8(ascii: ",")
  static let jsonColon = UInt8(ascii: ":")
  static let jsonQuote = UInt8(ascii: "\"")
  static let jsonEscape = UInt8(ascii: "\\")
  static let jsonBackspace = UInt8(ascii: "\u{08}")
  static let jsonFormFeed = UInt8(ascii: "\u{0C}")
  static let jsonNewline = UInt8(ascii: "\n")
  static let jsonCarriageReturn = UInt8(ascii: "\r")
  static let jsonTab = UInt8(ascii: "\t")
  static let jsonControlCharacterLimit = UInt8(ascii: " ")

  var isASCIIWhitespace: Bool {
    self == UInt8(ascii: " ") || self == UInt8(ascii: "\n")
      || self == UInt8(ascii: "\r") || self == UInt8(ascii: "\t")
  }
}
