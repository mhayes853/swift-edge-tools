#if FoundationModels && canImport(FoundationModels)
  public struct EdgeToolsFMConversionError: Error, Hashable, Sendable {
    public let message: String

    public static func nonFiniteNumber(_ value: Double) -> Self {
      Self(message: "FoundationModels cannot represent the nonfinite number \(value).")
    }

    public static func missingSchemaName(path: String) -> Self {
      Self(message: "A FoundationModels schema at \(path) requires a name.")
    }

    public static func malformedSchema(path: String, description: String) -> Self {
      Self(message: "Malformed schema at \(path): \(description)")
    }

    public static func unsupportedDynamicSchema(path: String, keyword: String) -> Self {
      Self(message: "DynamicGenerationSchema does not support '\(keyword)' at \(path).")
    }

    public static func invalidGenerationSchema(description: String) -> Self {
      Self(message: "Invalid FoundationModels generation schema: \(description)")
    }
  }
#endif
