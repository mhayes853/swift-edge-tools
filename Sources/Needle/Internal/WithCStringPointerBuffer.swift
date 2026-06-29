func withCopiedCStringPointerBuffer<Result>(
  _ strings: [String],
  _ body: (UnsafeMutableBufferPointer<UnsafePointer<CChar>?>) throws -> Result
) rethrows -> Result {
  var mutablePointers = strings.map { UnsafePointer<CChar>(strdup($0.utf8CString)) }
  defer { mutablePointers.forEach { $0?.deallocate() } }
  return try mutablePointers.withUnsafeMutableBufferPointer { try body($0) }
}

private func strdup(_ s: ContiguousArray<CChar>) -> UnsafeMutablePointer<CChar>? {
  let dst = UnsafeMutablePointer<CChar>.allocate(capacity: s.count)
  for i in 0..<s.count {
    dst[i] = s[i]
  }
  return dst
}
