extension UInt8 {
  var isASCIIWhitespace: Bool {
    self == UInt8(ascii: " ") || self == UInt8(ascii: "\n")
      || self == UInt8(ascii: "\r") || self == UInt8(ascii: "\t")
  }
}
