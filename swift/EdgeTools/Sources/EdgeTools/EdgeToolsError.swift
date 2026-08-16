import EdgeToolsCore

// MARK: - EdgeToolsError

extension EdgeToolsError.Code {
  public static let contextLengthExceeded = Self(rawValue: "context-length-exceeded")
  public static let failedToLoadConfiguration = Self(rawValue: "failed-to-load-configuration")
  public static let missingModelWeights = Self(rawValue: "missing-model-weights")
  public static let unsupportedTokenizer = Self(rawValue: "unsupported-tokenizer")
  public static let grammarRejectedToken = Self(rawValue: "grammar-rejected-token")
  public static let missingModelOutputs = Self(rawValue: "missing-model-outputs")
  public static let modelNotPrepared = Self(rawValue: "model-not-prepared")
  public static let invalidMedia = Self(rawValue: "invalid-media")
  public static let unsupportedMedia = Self(rawValue: "unsupported-media")
  public static let contextInUse = Self(rawValue: "context-in-use")
  public static let incompatibleContext = Self(rawValue: "incompatible-context")
}

extension EdgeToolsError {
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
  static let missingModelWeights = Self(
    code: .missingModelWeights,
    message: "No safetensor model weights were found."
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

  static func invalidMedia(_ message: String) -> Self {
    Self(code: .invalidMedia, message: message)
  }

  static func unsupportedMedia(_ message: String) -> Self {
    Self(code: .unsupportedMedia, message: message)
  }

  static let contextInUse = Self(
    code: .contextInUse,
    message: "The context already has an active generation."
  )

  static let incompatibleContext = Self(
    code: .incompatibleContext,
    message: "The context was created by a different engine."
  )
}
