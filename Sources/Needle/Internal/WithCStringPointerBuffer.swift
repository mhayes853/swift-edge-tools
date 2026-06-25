func withCopiedCStringPointerBuffer<Result>(
  _ strings: [String],
  _ body: (UnsafeMutableBufferPointer<UnsafePointer<CChar>?>) throws -> Result
) rethrows -> Result {
  let mutablePointers = strings.map { strdup($0) }
  defer { mutablePointers.forEach { $0?.deallocate() } }

  var constPointers = mutablePointers.map { UnsafePointer<CChar>($0) }
  return try constPointers.withUnsafeMutableBufferPointer { buffer in
    try body(buffer)
  }
}
