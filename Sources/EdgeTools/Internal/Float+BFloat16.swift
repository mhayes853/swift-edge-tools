extension Float {
  init(bfloat16Bits bits: UInt16) {
    self.init(bitPattern: UInt32(bits) << 16)
  }
}
