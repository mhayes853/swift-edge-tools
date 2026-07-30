// MARK: - EdgeToolsError

public struct EdgeToolsError: Error, Hashable, Sendable {
  public struct Code: RawRepresentable, Hashable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
      self.rawValue = rawValue
    }

    public static let noCompatibleTokenizer = Self(rawValue: "no-compatible-tokenizer")
    public static let unsupportedTransformersTokenizer = Self(
      rawValue: "unsupported-transformers-tokenizer"
    )
    public static let invalidHuggingFaceBackendJSON = Self(
      rawValue: "invalid-hugging-face-backend-json"
    )
    public static let emptyJSONInput = Self(rawValue: "empty-json-input")
    public static let invalidJSON = Self(rawValue: "invalid-json")
    public static let jsonIntegerOutOfRange = Self(rawValue: "json-integer-out-of-range")
    public static let nonFiniteJSONNumber = Self(rawValue: "non-finite-json-number")
    public static let invalidJSONValue = Self(rawValue: "invalid-json-value")
    public static let contextLengthExceeded = Self(rawValue: "context-length-exceeded")
    public static let failedToLoadConfiguration = Self(rawValue: "failed-to-load-configuration")
    public static let unsupportedTokenizer = Self(rawValue: "unsupported-tokenizer")
    public static let grammarRejectedToken = Self(rawValue: "grammar-rejected-token")
    public static let missingModelOutputs = Self(rawValue: "missing-model-outputs")
    public static let modelNotPrepared = Self(rawValue: "model-not-prepared")
  }

  public let code: Code
  public let message: String

  public init(code: Code, message: String) {
    self.code = code
    self.message = message
  }
}

extension EdgeToolsError {
  static func noCompatibleTokenizer(
    in directory: String,
    hasSentencePieceModel: Bool,
    hasTransformersTokenizer: Bool,
    failures: [String] = []
  ) -> Self {
    var details = [String]()

    if hasSentencePieceModel {
      details.append("tokenizer.model is not a supported SentencePiece BPE model.")
    } else {
      details.append("tokenizer.model was not found.")
    }

    #if Transformers
      if hasTransformersTokenizer {
        details.append("tokenizer.json is not a supported Transformers tokenizer.")
      } else {
        details.append("tokenizer.json was not found.")
      }
    #else
      if hasTransformersTokenizer {
        details.append(
          "tokenizer.json exists, but the Transformers trait is not enabled. Enable Transformers to load it."
        )
      } else {
        details.append(
          "The Transformers trait is not enabled; enable it to search for tokenizer.json."
        )
      }
    #endif

    details.append(contentsOf: failures)
    return Self(
      code: .noCompatibleTokenizer,
      message:
        "No compatible tokenizer was found in \(directory). \(details.joined(separator: " "))"
    )
  }

  static let invalidHuggingFaceBackendJSON = Self(
    code: .invalidHuggingFaceBackendJSON,
    message: "Invalid Hugging Face tokenizer JSON."
  )

  static let emptyJSONInput = Self(code: .emptyJSONInput, message: "Expected JSON input.")
  static let invalidJSONValue = Self(code: .invalidJSONValue, message: "Invalid JSON value.")

  static func contextLengthExceeded(tokens: Int, maximum: Int) -> Self {
    Self(
      code: .contextLengthExceeded,
      message: "Prompt token count (\(tokens)) exceeds the model context length (\(maximum))."
    )
  }

  static let failedToLoadConfiguration = Self(
    code: .failedToLoadConfiguration,
    message: "Could not load model configuration."
  )
  static let unsupportedTokenizer = Self(
    code: .unsupportedTokenizer,
    message: "The model does not support the provided tokenizer."
  )
  static func grammarRejectedToken(token: EdgeToolsToken) -> Self {
    Self(
      code: .grammarRejectedToken,
      message:
        "Token (ID=\(token.id), VALUE=\(token.stringValue)) was rejected by the grammar matcher."
    )
  }

  static let missingModelOutputs = Self(
    code: .missingModelOutputs,
    message: "The model did not return expected outputs."
  )

  static let modelNotPrepared = Self(
    code: .modelNotPrepared,
    message: "The model has no active generation. Call prepare before decoding."
  )
}
