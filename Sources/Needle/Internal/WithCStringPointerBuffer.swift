func withCopiedCStringPointerBuffer<Result>(
  _ strings: [String],
  _ body: (UnsafeMutableBufferPointer<UnsafePointer<CChar>?>) throws -> Result
) rethrows -> Result {
  var mutablePointers = strings.map {
    let dst = UnsafeMutablePointer<CChar>.allocate(capacity: $0.utf8CString.count)
    for i in 0..<$0.utf8CString.count {
      dst[i] = $0.utf8CString[i]
    }
    return UnsafePointer<CChar>(dst) as UnsafePointer<CChar>?
  }
  defer { mutablePointers.forEach { $0?.deallocate() } }
  return try mutablePointers.withUnsafeMutableBufferPointer { try body($0) }
}
