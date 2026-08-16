// MARK: - EdgeToolsTokenizerError

public struct EdgeToolsTokenizerError: Error, Hashable, Sendable {
  public struct Code: RawRepresentable, Hashable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
      self.rawValue = rawValue
    }

    public static let nativeFailure = Self(rawValue: "native-failure")
    public static let invalidBackendJSON = Self(rawValue: "invalid-backend-json")
    public static let missingChatTemplate = Self(rawValue: "missing-chat-template")
  }

  public let code: Code
  public let message: String

  public init(code: Code, message: String) {
    self.code = code
    self.message = message
  }
}

extension EdgeToolsTokenizerError {
  static let invalidBackendJSON = Self(
    code: .invalidBackendJSON,
    message: "Invalid Hugging Face tokenizer JSON."
  )
}
