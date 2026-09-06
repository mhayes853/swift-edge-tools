#if FoundationModels && canImport(FoundationModels)
  import EdgeToolsCore
  import _EdgeToolsFoundation
  import FoundationModels
  import OrderedCollections
#endif

// MARK: - Foundation Models Conversion Errors

#if FoundationModels && canImport(FoundationModels)
  public struct FMConversionError: Error, Hashable, Sendable {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let nonFiniteNumber = Self(rawValue: "non-finite-number")
      public static let malformedSchema = Self(rawValue: "malformed-schema")
      public static let invalidGenerationSchema = Self(rawValue: "invalid-generation-schema")
    }

    public let code: Code
    public let message: String

    public init(code: Code, message: String) {
      self.code = code
      self.message = message
    }

    static func invalidGenerationSchema(description: String) -> Self {
      Self(
        code: .invalidGenerationSchema,
        message: "Invalid FoundationModels generation schema: \(description)"
      )
    }
  }

#endif

// MARK: - Generated Content Conversion

#if FoundationModels && canImport(FoundationModels)
  // MARK: - EdgeToolsValue

  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  extension EdgeToolsValue {
    public init(generatedContent: GeneratedContent) throws {
      switch generatedContent.kind {
      case .null:
        self = .null
      case .bool(let value):
        self = .boolean(value)
      case .number(let value):
        self = .number(value)
      case .string(let value):
        self = .string(value)
      case .array(let values):
        self = try .array(values.map { try Self(generatedContent: $0) })
      case .structure(let properties, let orderedKeys):
        let remainingKeys = properties.keys.filter { !orderedKeys.contains($0) }.sorted()
        self = try .object(
          OrderedDictionary(
            uniqueKeysWithValues: (orderedKeys + remainingKeys)
              .compactMap { key in
                try properties[key].map { (key, try Self(generatedContent: $0)) }
              }
          )
        )
      @unknown default:
        throw FMConversionError(
          code: .malformedSchema,
          message: "Unsupported generated content kind."
        )
      }
    }
  }

  // MARK: - GeneratedContent

  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  extension GeneratedContent: ConvertibleFromEdgeToolsValue {
    public init(edgeToolsValue: EdgeToolsValue) throws {
      let kind: GeneratedContent.Kind =
        switch edgeToolsValue {
        case .null:
          .null
        case .boolean(let value):
          .bool(value)
        case .integer(let value):
          .number(Double(value))
        case .number(let value):
          if value.isFinite {
            .number(value)
          } else {
            throw FMConversionError(
              code: .nonFiniteNumber,
              message: "FoundationModels cannot represent the nonfinite number \(value)."
            )
          }
        case .string(let value):
          .string(value)
        case .array(let values):
          .array(try values.map { try GeneratedContent(edgeToolsValue: $0) })
        case .object(let properties):
          .structure(
            properties: try Dictionary(
              uniqueKeysWithValues: properties.map { key, value in
                (key, try GeneratedContent(edgeToolsValue: value))
              }
            ),
            orderedKeys: Array(properties.keys)
          )
        }
      self.init(kind: kind)
    }
  }
#endif

// MARK: - Foundation Models Schema Conversion

#if FoundationModels && canImport(FoundationModels)
  // MARK: - EdgeToolsGenerationSchema

  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  extension EdgeToolsGenerationSchema {
    public init(generationSchema: GenerationSchema) throws {
      do {
        let data = try JSONEncoder().encode(generationSchema)
        self = try JSONDecoder().decode(Self.self, from: data)
      } catch {
        throw FMConversionError.invalidGenerationSchema(
          description: String(describing: error)
        )
      }
    }
  }
#endif
