import Foundation

func withCStringPointerBuffer<Result>(
  _ strings: [String],
  _ body: (UnsafeMutableBufferPointer<UnsafePointer<CChar>?>) throws -> Result
) rethrows -> Result {
  var pointers = [UnsafePointer<CChar>?](repeating: nil, count: strings.count)

  func recurse(_ index: Int) throws -> Result {
    if index == strings.count {
      return try pointers.withUnsafeMutableBufferPointer { buffer in
        try body(buffer)
      }
    }
    return try strings[index]
      .withCString { cString in
        pointers[index] = cString
        return try recurse(index + 1)
      }
  }

  return try recurse(0)
}
