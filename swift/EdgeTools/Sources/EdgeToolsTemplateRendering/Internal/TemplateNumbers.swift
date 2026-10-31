// MARK: - JSON Numbers

func templateJSONNumber(_ value: Double) -> String {
  guard value.isFinite else {
    return "null"
  }
  guard value != 0 else {
    return value.sign == .minus ? "-0.0" : "0.0"
  }
  let (digits, exponent) = shortestDecimalDigits(value.magnitude)
  let sign = value.sign == .minus ? "-" : ""
  let length = digits.count
  let point = length + exponent
  let text = { (range: Range<Int>) in String(decoding: digits[range], as: UTF8.self) }
  if length <= point && point <= maxPositionalExponent {
    return sign + text(0..<length) + String(repeating: "0", count: point - length) + ".0"
  }
  if 0 < point && point <= maxPositionalExponent {
    return sign + text(0..<point) + "." + text(point..<length)
  }
  if minPositionalExponent < point && point <= 0 {
    return sign + "0." + String(repeating: "0", count: -point) + text(0..<length)
  }
  let mantissa = length == 1 ? text(0..<1) : text(0..<1) + "." + text(1..<length)
  let decimalExponent = point - 1
  let exponentDigits = "\(decimalExponent.magnitude)"
  return sign + mantissa + "e" + (decimalExponent < 0 ? "-" : "+")
    + (exponentDigits.utf8.count < 2 ? "0" : "") + exponentDigits
}

// MARK: - Fixed Notation

func templateFixedNotation(_ value: Double) -> String {
  if value.isNaN {
    return value.sign == .minus ? "-nan" : "nan"
  }
  if value.isInfinite {
    return value.sign == .minus ? "-inf" : "inf"
  }
  let magnitude = value.magnitude
  let exponent = magnitude.isNormal ? Int(magnitude.exponentBitPattern) - 1075 : -1074
  let significand = magnitude.significandBitPattern | (magnitude.isNormal ? 1 << 52 : 0)
  var limbs = [UInt32(significand % 1_000_000_000), UInt32(significand / 1_000_000_000)]
  for _ in 0..<exponent.magnitude {
    multiply(&limbs, by: exponent < 0 ? 5 : 2)
  }
  var digits = decimalDigits(limbs)
  let fractionDigitCount = max(-exponent, 0)
  if digits.count <= fractionDigitCount {
    digits.insert(contentsOf: repeatElement(0, count: fractionDigitCount - digits.count + 1), at: 0)
  }
  if fractionDigitCount > 6 {
    let dropped = digits.suffix(fractionDigitCount - 6)
    digits.removeLast(dropped.count)
    let roundingDigit = dropped.first ?? 0
    let isTie = roundingDigit == 5 && !dropped.dropFirst().contains { $0 != 0 }
    if roundingDigit > 5 || (roundingDigit == 5 && (!isTie || (digits.last ?? 0) % 2 == 1)) {
      roundUp(&digits)
    }
  } else {
    digits += repeatElement(0, count: 6 - fractionDigitCount)
  }
  let text = digits.map { Character(Unicode.Scalar($0 + 0x30)) }
  return (value.sign == .minus ? "-" : "") + String(text.dropLast(6)) + "."
    + String(text.suffix(6))
}

// MARK: - Integer Conversion

func templateTruncatedInteger(_ value: Double) -> Int64 {
  guard !value.isNaN else {
    return 0
  }
  return Int64(exactly: value.rounded(.towardZero)) ?? (value < 0 ? .min : .max)
}

func templateLeadingInteger(_ string: String) -> Int64 {
  var bytes = string.utf8.drop { $0.isTemplateSpace }
  let isNegative = bytes.first == UInt8(ascii: "-")
  if isNegative || bytes.first == UInt8(ascii: "+") {
    bytes = bytes.dropFirst()
  }
  let digits = bytes.prefix { (0x30...0x39).contains($0) }
  var magnitude: UInt64 = 0
  for digit in digits {
    let (product, overflowsProduct) = magnitude.multipliedReportingOverflow(by: 10)
    let (sum, overflowsSum) = product.addingReportingOverflow(UInt64(digit - 0x30))
    guard !overflowsProduct && !overflowsSum else {
      return 0
    }
    magnitude = sum
  }
  let limit = isNegative ? UInt64(Int64.max) + 1 : UInt64(Int64.max)
  guard magnitude <= limit else {
    return 0
  }
  return isNegative ? Int64(truncatingIfNeeded: 0 &- magnitude) : Int64(magnitude)
}

// MARK: - Helpers

private let minPositionalExponent = -4
private let maxPositionalExponent = 15

private func shortestDecimalDigits(_ magnitude: Double) -> (digits: [UInt8], exponent: Int) {
  let description = Array(magnitude.description.utf8)
  var digits = [UInt8]()
  var exponent = 0
  var fractionDigitCount = 0
  var isInFraction = false
  for (index, byte) in description.enumerated() {
    if byte == UInt8(ascii: ".") {
      isInFraction = true
    } else if byte == UInt8(ascii: "e") || byte == UInt8(ascii: "E") {
      exponent = Int(String(decoding: description[(index + 1)...], as: UTF8.self)) ?? 0
      break
    } else {
      digits.append(byte)
      fractionDigitCount += isInFraction ? 1 : 0
    }
  }
  exponent -= fractionDigitCount
  digits.removeFirst(digits.prefix { $0 == UInt8(ascii: "0") }.count)
  while digits.last == UInt8(ascii: "0") {
    digits.removeLast()
    exponent += 1
  }
  return (digits, exponent)
}

private func multiply(_ limbs: inout [UInt32], by factor: UInt64) {
  var carry: UInt64 = 0
  for index in limbs.indices {
    let product = UInt64(limbs[index]) * factor + carry
    limbs[index] = UInt32(product % 1_000_000_000)
    carry = product / 1_000_000_000
  }
  if carry > 0 {
    limbs.append(UInt32(carry))
  }
}

private func decimalDigits(_ limbs: [UInt32]) -> [UInt8] {
  let text = limbs.reversed()
    .map { limb in
      let digits = "\(limb)"
      return String(repeating: "0", count: 9 - digits.utf8.count) + digits
    }
    .joined()
  return text.utf8.drop { $0 == UInt8(ascii: "0") }.map { $0 - UInt8(ascii: "0") }
}

private func roundUp(_ digits: inout [UInt8]) {
  var index = digits.count - 1
  while index >= 0 && digits[index] == 9 {
    digits[index] = 0
    index -= 1
  }
  if index >= 0 {
    digits[index] += 1
  } else {
    digits.insert(1, at: 0)
  }
}
