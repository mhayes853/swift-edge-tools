import CustomDump
import Testing

func expectNoDifference<T: Equatable>(
  _ first: @autoclosure () throws -> T,
  _ second: @autoclosure () throws -> T,
  _ message: @autoclosure () -> String? = nil,
  fileID: StaticString = #fileID,
  filePath: StaticString = #filePath,
  line: UInt = #line,
  column: UInt = #column
) {
  #if os(WASI)
    // NB: isTesting is false for IssueReporting for WASI, so we need to reimplement with an
    // explicit Issue.record.
    let sourceLocation = SourceLocation(
      fileID: "\(fileID)",
      filePath: "\(filePath)",
      line: Int(line),
      column: Int(column)
    )
    do {
      let first = try first()
      let second = try second()
      guard first != second else { return }

      let difference = CustomDump.diff(first, second, format: .proportional)
        ?? "(\(first)) is not equal to (\(second)), but no difference was detected."
      let indentedDifference = difference
        .split(separator: "\n", omittingEmptySubsequences: false)
        .map { "  \($0)" }
        .joined(separator: "\n")
      let prefix = message().map { "\($0) - " } ?? ""
      Issue.record(
        Comment(
          rawValue: """
            \(prefix)Difference: …

            \(indentedDifference)

            (First: −, Second: +)
            """
        ),
        sourceLocation: sourceLocation
      )
    } catch {
      Issue.record(error, sourceLocation: sourceLocation)
    }
  #else
    CustomDump.expectNoDifference(
      try first(),
      try second(),
      message(),
      fileID: fileID,
      filePath: filePath,
      line: line,
      column: column
    )
  #endif
}
