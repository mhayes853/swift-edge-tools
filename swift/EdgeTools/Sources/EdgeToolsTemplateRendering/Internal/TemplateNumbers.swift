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
  var scaled = BigUnsignedInteger(
    magnitude.significandBitPattern | (magnitude.isNormal ? 1 << 52 : 0)
  )
  scaled.multiply(by: 1_000_000)
  if exponent >= 0 {
    scaled.shiftLeft(by: exponent)
  } else {
    scaled.shiftRightRoundingToEven(by: -exponent)
  }
  var digits = scaled.decimalDigits()
  if digits.count < 7 {
    digits.insert(contentsOf: repeatElement(UInt8(ascii: "0"), count: 7 - digits.count), at: 0)
  }
  digits.insert(UInt8(ascii: "."), at: digits.count - 6)
  return (value.sign == .minus ? "-" : "") + String(decoding: digits, as: UTF8.self)
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

private struct BigUnsignedInteger {
  private var limbs: [UInt32]

  init(_ value: UInt64) {
    self.limbs = [UInt32(truncatingIfNeeded: value), UInt32(truncatingIfNeeded: value >> 32)]
  }

  mutating func multiply(by factor: UInt32) {
    var carry: UInt64 = 0
    for index in self.limbs.indices {
      let product = UInt64(self.limbs[index]) * UInt64(factor) + carry
      self.limbs[index] = UInt32(truncatingIfNeeded: product)
      carry = product >> 32
    }
    if carry > 0 {
      self.limbs.append(UInt32(carry))
    }
  }

  mutating func shiftLeft(by bitCount: Int) {
    let bitShift = bitCount % 32
    if bitShift > 0 {
      var carry: UInt32 = 0
      for index in self.limbs.indices {
        let limb = self.limbs[index]
        self.limbs[index] = (limb << bitShift) | carry
        carry = limb >> (32 - bitShift)
      }
      self.limbs.append(carry)
    }
    self.limbs.insert(contentsOf: repeatElement(0, count: bitCount / 32), at: 0)
  }

  mutating func shiftRightRoundingToEven(by bitCount: Int) {
    let isAboveHalf = self.bit(at: bitCount - 1)
    let hasRemainder = self.hasSetBit(below: bitCount - 1)
    self.shiftRight(by: bitCount)
    if isAboveHalf && (hasRemainder || self.limbs[0] & 1 == 1) {
      self.increment()
    }
  }

  func decimalDigits() -> [UInt8] {
    var remaining = self.limbs
    var chunks = [UInt32]()
    while remaining.contains(where: { $0 != 0 }) {
      var remainder: UInt64 = 0
      for index in remaining.indices.reversed() {
        let current = (remainder << 32) | UInt64(remaining[index])
        remaining[index] = UInt32(current / 1_000_000_000)
        remainder = current % 1_000_000_000
      }
      chunks.append(UInt32(remainder))
    }
    guard let leading = chunks.popLast() else {
      return [UInt8(ascii: "0")]
    }
    return chunks.reversed()
      .reduce(into: Array("\(leading)".utf8)) { digits, chunk in
        let chunkDigits = Array("\(chunk)".utf8)
        digits += repeatElement(UInt8(ascii: "0"), count: 9 - chunkDigits.count)
        digits += chunkDigits
      }
  }

  private mutating func shiftRight(by bitCount: Int) {
    guard bitCount / 32 < self.limbs.count else {
      self.limbs = [0]
      return
    }
    self.limbs.removeFirst(bitCount / 32)
    let bitShift = bitCount % 32
    guard bitShift > 0 else {
      return
    }
    for index in self.limbs.indices {
      let next = index + 1 < self.limbs.count ? self.limbs[index + 1] : 0
      self.limbs[index] = (self.limbs[index] >> bitShift) | (next << (32 - bitShift))
    }
  }

  private mutating func increment() {
    for index in self.limbs.indices {
      self.limbs[index] &+= 1
      if self.limbs[index] != 0 {
        return
      }
    }
    self.limbs.append(1)
  }

  private func bit(at index: Int) -> Bool {
    guard index / 32 < self.limbs.count else {
      return false
    }
    return (self.limbs[index / 32] >> (index % 32)) & 1 == 1
  }

  private func hasSetBit(below index: Int) -> Bool {
    let wholeLimbs = min(index / 32, self.limbs.count)
    if self.limbs.prefix(wholeLimbs).contains(where: { $0 != 0 }) {
      return true
    }
    guard index / 32 < self.limbs.count, index % 32 > 0 else {
      return false
    }
    return self.limbs[index / 32] & ((1 << (index % 32)) - 1) != 0
  }
}
