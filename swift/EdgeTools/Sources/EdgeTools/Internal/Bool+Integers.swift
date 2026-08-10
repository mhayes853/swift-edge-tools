extension Bool {
  func intValue<I: BinaryInteger>(as type: I.Type) -> I {
    self ? 1 : 0
  }
}

extension BinaryInteger {
  @usableFromInline
  var boolValue: Bool {
    self == 0 ? false : true
  }
}
