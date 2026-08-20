// MARK: - EdgeToolsDescribableError

public protocol EdgeToolsDescribableError: Error {
  var edgeToolsErrorDescription: String { get }
}

// MARK: - EdgeToolsError

public struct EdgeToolsError: Error, Hashable, Sendable {
  public struct Code: RawRepresentable, Hashable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
      self.rawValue = rawValue
    }

    public static let noCompatibleTokenizer = Self(rawValue: "no-compatible-tokenizer")
    public static let emptyJSONInput = Self(rawValue: "empty-json-input")
    public static let invalidJSON = Self(rawValue: "invalid-json")
    public static let jsonIntegerOutOfRange = Self(rawValue: "json-integer-out-of-range")
    public static let nonFiniteJSONNumber = Self(rawValue: "non-finite-json-number")
    public static let invalidJSONValue = Self(rawValue: "invalid-json-value")
    public static let toolInvocationFailed = Self(rawValue: "tool-invocation-failed")
  }

  public let code: Code
  public let message: String

  public init(code: Code, message: String) {
    self.code = code
    self.message = message
  }
}

extension EdgeToolsError: EdgeToolsDescribableError {
  public var edgeToolsErrorDescription: String { self.message }
}

extension EdgeToolsError {
  static let emptyJSONInput = Self(code: .emptyJSONInput, message: "Expected JSON input.")
  static let invalidJSONValue = Self(code: .invalidJSONValue, message: "Invalid JSON value.")

  static let jsonIntegerOutOfRange = Self(
    code: .jsonIntegerOutOfRange,
    message: "A JSON integer was out of the representable range."
  )

  static let nonFiniteJSONNumber = Self(
    code: .nonFiniteJSONNumber,
    message: "A JSON number was not finite."
  )
}
