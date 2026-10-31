// MARK: - Bytes

extension UInt8 {
  static let newline = Self(ascii: "\n")

  var isTemplateSpace: Bool {
    self == 0x20 || (0x09...0x0D).contains(self)
  }

  var isTemplateWord: Bool {
    self.isASCIILetter || self.isASCIIDigit || self == UInt8(ascii: "_")
  }

  var isASCIIDigit: Bool {
    (0x30...0x39).contains(self)
  }

  var isASCIILetter: Bool {
    (0x41...0x5A).contains(self) || (0x61...0x7A).contains(self)
  }

  var asciiUppercased: Self {
    (0x61...0x7A).contains(self) ? self - 0x20 : self
  }

  var asciiLowercased: Self {
    (0x41...0x5A).contains(self) ? self + 0x20 : self
  }
}

// MARK: - String Operations

func templateStrip(_ string: String, characters: String = "", left: Bool = true, right: Bool = true)
  -> String
{
  let characters = Set((characters.isEmpty ? " \t\n\r" : characters).unicodeScalars)
  let scalars = string.unicodeScalars
  let start = left ? scalars.firstIndex { !characters.contains($0) } : scalars.startIndex
  guard let start else {
    return ""
  }
  let end = right ? scalars.lastIndex { !characters.contains($0) } : scalars.indices.last
  guard let end, start <= end else {
    return ""
  }
  return String(scalars[start...end])
}

func templateSplit(_ string: String, separator: String) throws(TemplateFailure) -> [String] {
  guard !separator.isEmpty else {
    throw TemplateFailure("empty separator")
  }
  let bytes = Array(string.utf8)
  let separatorBytes = Array(separator.utf8)
  var parts = [String]()
  var start = 0
  while let match = templateFind(separatorBytes, in: bytes, from: start) {
    parts.append(String(decoding: bytes[start..<match], as: UTF8.self))
    start = match + separatorBytes.count
  }
  parts.append(String(decoding: bytes[start...], as: UTF8.self))
  return parts
}

func templateReplace(_ string: String, _ before: String, _ after: String, count: Int64?) -> String {
  var bytes = Array(string.utf8)
  let beforeBytes = Array(before.utf8)
  let afterBytes = Array(after.utf8)
  var remaining = count.map { $0 < 0 ? Int64.max : $0 } ?? Int64(bytes.count)
  var start = 0
  while remaining > 0, let match = templateFind(beforeBytes, in: bytes, from: start) {
    bytes.replaceSubrange(match..<(match + beforeBytes.count), with: afterBytes)
    start = match + afterBytes.count
    remaining -= 1
  }
  return String(decoding: bytes, as: UTF8.self)
}

func templateContains(_ string: String, _ substring: String) -> Bool {
  templateFind(Array(substring.utf8), in: Array(string.utf8), from: 0) != nil
}

func templateHasPrefix(_ string: String, _ prefix: String) -> Bool {
  string.utf8.starts(with: prefix.utf8)
}

func templateHasSuffix(_ string: String, _ suffix: String) -> Bool {
  string.utf8.reversed().starts(with: suffix.utf8.reversed())
}

func templateIsLess(_ lhs: String, _ rhs: String) -> Bool {
  lhs.utf8.lexicographicallyPrecedes(rhs.utf8)
}

func templateUppercased(_ string: String) -> String {
  String(decoding: string.utf8.map(\.asciiUppercased), as: UTF8.self)
}

func templateLowercased(_ string: String) -> String {
  String(decoding: string.utf8.map(\.asciiLowercased), as: UTF8.self)
}

func templateCapitalized(_ string: String) -> String {
  var bytes = Array(string.utf8)
  if let first = bytes.first {
    bytes[0] = first.asciiUppercased
  }
  return String(decoding: bytes, as: UTF8.self)
}

func templateTitleCased(_ string: String) -> String {
  var bytes = Array(string.utf8)
  for index in bytes.indices {
    let startsWord = index == 0 || bytes[index - 1].isTemplateSpace
    bytes[index] = startsWord ? bytes[index].asciiUppercased : bytes[index].asciiLowercased
  }
  return String(decoding: bytes, as: UTF8.self)
}

func templateHTMLEscaped(_ string: String) -> String {
  var escaped = ""
  escaped.reserveCapacity(string.utf8.count)
  for scalar in string.unicodeScalars {
    switch scalar {
    case "&": escaped += "&amp;"
    case "<": escaped += "&lt;"
    case ">": escaped += "&gt;"
    case "\"": escaped += "&#34;"
    case "'": escaped += "&apos;"
    default: escaped.unicodeScalars.append(scalar)
    }
  }
  return escaped
}

func templateIndented(_ text: String, width: Int64, indentsFirstLine: Bool) -> String {
  var lines = text.unicodeScalars.split(separator: "\n", omittingEmptySubsequences: false)
  if text.utf8.isEmpty || text.utf8.last == .newline {
    lines.removeLast()
  }
  let indentation = String(repeating: " ", count: Int(clamping: max(width, 0)))
  var indented = lines.enumerated()
    .map { index, line in (index > 0 || indentsFirstLine ? indentation : "") + String(line) }
    .joined(separator: "\n")
  if text.utf8.last == .newline {
    indented += "\n"
  }
  return indented
}

// MARK: - Quoting

func appendJSONString(_ string: String, ensuringASCII: Bool, to output: inout String) {
  output += "\""
  for scalar in string.unicodeScalars {
    switch scalar.value {
    case 0x22: output += "\\\""
    case 0x5C: output += "\\\\"
    case 0x08: output += "\\b"
    case 0x0C: output += "\\f"
    case 0x0A: output += "\\n"
    case 0x0D: output += "\\r"
    case 0x09: output += "\\t"
    case 0..<0x20:
      appendUnicodeEscape(UInt16(scalar.value), to: &output)
    case 0x10000... where ensuringASCII:
      let offset = scalar.value - 0x10000
      appendUnicodeEscape(UInt16(0xD800 + (offset >> 10)), to: &output)
      appendUnicodeEscape(UInt16(0xDC00 + (offset & 0x3FF)), to: &output)
    case 0x80... where ensuringASCII:
      appendUnicodeEscape(UInt16(scalar.value), to: &output)
    default:
      output.unicodeScalars.append(scalar)
    }
  }
  output += "\""
}

func appendPythonString(_ string: String, to output: inout String) {
  var json = ""
  appendJSONString(string, ensuringASCII: false, to: &json)
  guard !string.utf8.contains(UInt8(ascii: "'")) else {
    output += json
    return
  }
  let bytes = Array(json.utf8)
  var quoted = [UInt8(ascii: "'")]
  var index = 1
  while index < bytes.count - 1 {
    if bytes[index] == UInt8(ascii: "\\") && bytes[index + 1] == UInt8(ascii: "\"") {
      quoted.append(UInt8(ascii: "\""))
      index += 2
    } else {
      quoted.append(bytes[index])
      index += 1
    }
  }
  quoted.append(UInt8(ascii: "'"))
  output += String(decoding: quoted, as: UTF8.self)
}

// MARK: - Helpers

private func templateFind(_ needle: [UInt8], in haystack: [UInt8], from start: Int) -> Int? {
  guard start <= haystack.count - needle.count else {
    return nil
  }
  return (start...(haystack.count - needle.count))
    .first { haystack[$0..<($0 + needle.count)].elementsEqual(needle) }
}

private func appendUnicodeEscape(_ unit: UInt16, to output: inout String) {
  let digits = Array("0123456789abcdef".utf8)
  output += "\\u"
  output += String(
    decoding: [12, 8, 4, 0].map { digits[Int((unit >> $0) & 0xF)] },
    as: UTF8.self
  )
}
