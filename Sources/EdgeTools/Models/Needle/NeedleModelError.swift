// MARK: - NeedleModelError

public struct NeedleModelError: Hashable, Error {
  public let message: String

  public static func contextLengthExceeded(tokens: Int, maximum: Int) -> Self {
    Self(message: "Prompt token count (\(tokens)) exceeds the model context length (\(maximum)).")
  }
}
