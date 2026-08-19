#if canImport(CLlama)
  package func llamaMeasuredCString(
    measure: () -> Int32,
    fill: (UnsafeMutablePointer<CChar>?, Int) -> Int32
  ) -> String? {
    let measured = measure()
    let count = measured >= 0 ? measured : -measured
    var storage = [CChar](repeating: 0, count: Int(count) + 1)
    let written = storage.withUnsafeMutableBufferPointer { fill($0.baseAddress, $0.count) }
    guard written >= 0 else { return nil }
    return storage.withUnsafeBufferPointer { buffer in
      String(decoding: buffer.prefix(Int(written)).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
  }
#endif
