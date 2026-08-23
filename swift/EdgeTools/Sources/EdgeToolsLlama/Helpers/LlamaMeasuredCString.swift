#if canImport(CLlama)
  package func llamaMeasuredCString(
    measure: () -> Int32,
    fill: (UnsafeMutablePointer<CChar>?, Int) -> Int32
  ) -> String? {
    let measured = measure()
    guard
      let count = Int(exactly: measured.magnitude),
      count < Int.max
    else {
      return nil
    }
    var storage = [CChar](repeating: 0, count: count + 1)
    let written = storage.withUnsafeMutableBufferPointer { fill($0.baseAddress, $0.count) }
    guard written >= 0 else { return nil }
    return storage.withUnsafeBufferPointer { buffer in
      String(decoding: buffer.prefix(Int(written)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
  }
#endif
