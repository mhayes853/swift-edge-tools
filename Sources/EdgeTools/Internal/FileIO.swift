import SystemPackage

#if canImport(Darwin)
  import Darwin
#elseif canImport(Glibc)
  import Glibc
#elseif canImport(Musl)
  import Musl
#endif

package func readFile(at path: FilePath) throws -> [UInt8] {
  let descriptor = try FileDescriptor.open(path, .readOnly)
  return try descriptor.closeAfter {
    var contents = [UInt8]()
    var buffer = [UInt8](repeating: 0, count: 64 * 1024)
    while true {
      let count = try buffer.withUnsafeMutableBytes { try descriptor.read(into: $0) }
      guard count > 0 else { return contents }
      contents.append(contentsOf: buffer.prefix(count))
    }
  }
}

package func withFileBytes<Result>(
  at path: FilePath,
  _ body: (UnsafeBufferPointer<UInt8>) throws -> Result
) throws -> Result {
  #if canImport(Darwin) || canImport(Glibc) || canImport(Musl)
    switch try withMemoryMappedFileBytes(at: path, body) {
    case .value(let result): return result
    case .unavailable: break
    }
  #endif

  let bytes = try readFile(at: path)
  return try bytes.withUnsafeBufferPointer(body)
}

package func fileExists(at path: FilePath) -> Bool {
  guard let descriptor = try? FileDescriptor.open(path, .readOnly) else { return false }
  return (try? descriptor.close()) != nil
}

#if canImport(Darwin) || canImport(Glibc) || canImport(Musl)
  private enum MemoryMappedFileResult<Result> {
    case unavailable
    case value(Result)
  }

  private func withMemoryMappedFileBytes<Result>(
    at path: FilePath,
    _ body: (UnsafeBufferPointer<UInt8>) throws -> Result
  ) throws -> MemoryMappedFileResult<Result> {
    guard let descriptor = try? FileDescriptor.open(path, .readOnly) else {
      return .unavailable
    }
    defer { try? descriptor.close() }

    guard let size = try? descriptor.stat().size,
      let count = Int(exactly: size),
      count >= 0
    else { return .unavailable }
    guard count > 0 else {
      return .value(try body(UnsafeBufferPointer(start: nil, count: 0)))
    }

    let mapping = mmap(nil, count, PROT_READ, MAP_PRIVATE, descriptor.rawValue, 0)
    guard mapping != MAP_FAILED, let mapping else { return .unavailable }
    defer { munmap(mapping, count) }

    let bytes = UnsafeRawBufferPointer(start: mapping, count: count).bindMemory(to: UInt8.self)
    return .value(try body(bytes))
  }
#endif
