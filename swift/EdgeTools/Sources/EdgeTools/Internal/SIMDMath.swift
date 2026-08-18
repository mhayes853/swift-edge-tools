#if canImport(simd)
  import simd
#endif

// MARK: - Exp

@inline(always)
func simdExpShifted(
  _ values: UnsafeBufferPointer<Float>,
  shiftedBy maximum: Float,
  scaledBy scale: Float,
  into destination: UnsafeMutableBufferPointer<Float>
) -> Float {
  guard let source = values.baseAddress, let target = destination.baseAddress else { return 0 }
  let width = SIMD16<Float>.scalarCount
  let shift = SIMD16<Float>(repeating: maximum)
  let factor = SIMD16<Float>(repeating: scale)
  var totals = SIMD16<Float>(repeating: 0)
  var index = 0
  while index + width <= values.count {
    let block = UnsafeRawPointer(source.advanced(by: index)).loadUnaligned(as: SIMD16<Float>.self)
    let weights = simdExp((block - shift) * factor)
    UnsafeMutableRawPointer(target.advanced(by: index))
      .storeBytes(of: weights, as: SIMD16<Float>.self)
    totals += weights
    index += width
  }
  var total = totals.sum()
  while index < values.count {
    let weight = expf((values[index] - maximum) * scale)
    destination[index] = weight
    total += weight
    index += 1
  }
  return total
}

@inline(always)
func simdExp(_ values: SIMD16<Float>) -> SIMD16<Float> {
  let magic = SIMD16<Float>(repeating: expRoundingMagic)
  let scaled = simdMax(values, SIMD16(repeating: smallestExpInput)) * log2Inverse
  let rounded = scaled + magic
  let fraction = scaled - (rounded - magic)
  var polynomial = SIMD16<Float>(repeating: 0.001_333_355_8)
  polynomial = polynomial * fraction + SIMD16(repeating: 0.009_618_129)
  polynomial = polynomial * fraction + SIMD16(repeating: 0.055_504_108_7)
  polynomial = polynomial * fraction + SIMD16(repeating: 0.240_226_506_9)
  polynomial = polynomial * fraction + SIMD16(repeating: 0.693_147_180_5)
  polynomial = polynomial * fraction + SIMD16(repeating: 1)
  let exponent = unsafeBitCast(rounded, to: SIMD16<UInt32>.self) &<< 23
  let bits = unsafeBitCast(polynomial, to: SIMD16<UInt32>.self) &+ exponent
  return unsafeBitCast(bits, to: SIMD16<Float>.self)
}

// MARK: - Min/Max

@inline(always)
func simdMax(_ lhs: SIMD16<Float>, _ rhs: SIMD16<Float>) -> SIMD16<Float> {
  #if canImport(simd)
    simd_max(lhs, rhs)
  #else
    lhs.replacing(with: rhs, where: rhs .> lhs)
  #endif
}

@inline(always)
func simdMin(_ lhs: SIMD16<Float>, _ rhs: SIMD16<Float>) -> SIMD16<Float> {
  #if canImport(simd)
    simd_min(lhs, rhs)
  #else
    lhs.replacing(with: rhs, where: rhs .< lhs)
  #endif
}

// MARK: - Constants

private let smallestExpInput = Float(-87.3)
private let log2Inverse = Float(1.442_695_04)
private let expRoundingMagic = Float(1 << 23) + Float(1 << 22)
