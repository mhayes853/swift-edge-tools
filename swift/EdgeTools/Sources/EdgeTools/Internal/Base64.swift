func base64EncodedString(_ bytes: [UInt8]) -> String {
  let alphabet = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/".utf8)
  var text = [UInt8]()
  text.reserveCapacity((bytes.count + 2) / 3 * 4)
  for index in stride(from: 0, to: bytes.count, by: 3) {
    let remaining = bytes.count - index
    let word = UInt32(bytes[index]) << 16
      | (remaining > 1 ? UInt32(bytes[index + 1]) << 8 : 0)
      | (remaining > 2 ? UInt32(bytes[index + 2]) : 0)
    text.append(alphabet[Int(word >> 18 & 63)])
    text.append(alphabet[Int(word >> 12 & 63)])
    text.append(remaining > 1 ? alphabet[Int(word >> 6 & 63)] : UInt8(ascii: "="))
    text.append(remaining > 2 ? alphabet[Int(word & 63)] : UInt8(ascii: "="))
  }
  return String(decoding: text, as: UTF8.self)
}
