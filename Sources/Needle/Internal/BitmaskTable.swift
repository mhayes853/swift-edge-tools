let bitmaskTable = {
  var values = [Float]()
  values.reserveCapacity(256 * 8)
  for byte in 0..<256 {
    for bit in 0..<8 {
      values.append(((byte >> bit) & 1) != 0 ? 0 : -.infinity)
    }
  }
  return values
}()
