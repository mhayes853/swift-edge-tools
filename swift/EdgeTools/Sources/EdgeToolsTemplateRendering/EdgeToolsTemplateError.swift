import EdgeToolsCore

// MARK: - EdgeToolsTemplateError

/// An error thrown while parsing or rendering an ``EdgeToolsTemplate``.
///
/// Messages for failures tied to a place in the template source end with the row and column of
/// that place, along with an excerpt of the surrounding lines.
public struct EdgeToolsTemplateError: Error, Hashable, Sendable {
  public struct Code: RawRepresentable, Hashable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
      self.rawValue = rawValue
    }

    /// The template source is not valid template syntax.
    public static let invalidSyntax = Self(rawValue: "invalid-syntax")

    /// The template failed while rendering against a context.
    public static let renderFailure = Self(rawValue: "render-failure")
  }

  public let code: Code
  public let message: String

  public init(code: Code, message: String) {
    self.code = code
    self.message = message
  }
}

extension EdgeToolsTemplateError: EdgeToolsDescribableError {
  public var edgeToolsErrorDescription: String { self.message }
}

// MARK: - TemplateFailure

struct TemplateFailure: Error {
  var message: String
  var position: Int?

  init(_ message: String, position: Int? = nil) {
    self.message = message
    self.position = position
  }

  func located(at position: Int) -> Self {
    Self(self.message, position: self.position ?? position)
  }

  func message(in source: [UInt8]) -> String {
    guard let position else {
      return self.message
    }
    return self.message + errorLocationSuffix(source, position)
  }
}

func withTemplateLocation<Value>(
  _ position: Int,
  _ body: () throws(TemplateFailure) -> Value
) throws(TemplateFailure) -> Value {
  do throws(TemplateFailure) {
    return try body()
  } catch {
    throw error.located(at: position)
  }
}

// MARK: - Error Location

private func errorLocationSuffix(_ source: [UInt8], _ position: Int) -> String {
  let position = min(max(position, 0), source.count)
  let lines = source.split(separator: .newline, omittingEmptySubsequences: false)
  let row = source[..<position].count { $0 == .newline } + 1
  let column = position - (source[..<position].lastIndex(of: .newline) ?? -1)
  var suffix = " at row \(row), column \(column):\n"
  if row > 1 {
    suffix += String(decoding: lines[row - 2], as: UTF8.self) + "\n"
  }
  suffix += String(decoding: lines[row - 1], as: UTF8.self) + "\n"
  suffix += String(repeating: " ", count: column - 1) + "^\n"
  if row < lines.count {
    suffix += String(decoding: lines[row], as: UTF8.self) + "\n"
  }
  return suffix
}
