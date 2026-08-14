func withOptionalCString<Result>(
  _ string: String?,
  body: (UnsafePointer<CChar>?) throws -> Result
) rethrows -> Result {
  guard let string else {
    return try body(nil)
  }
  return try string.withCString(body)
}
