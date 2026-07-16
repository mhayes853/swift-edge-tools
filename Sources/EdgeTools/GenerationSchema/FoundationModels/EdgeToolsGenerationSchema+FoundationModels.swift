#if FoundationModels && canImport(FoundationModels)
  import Foundation
  import FoundationModels
  import OrderedCollections

  // MARK: - EdgeToolsGenerationSchema

  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  extension EdgeToolsGenerationSchema {
    public init(generationSchema: GenerationSchema) throws {
      do {
        let data = try JSONEncoder().encode(generationSchema)
        self = try JSONDecoder().decode(Self.self, from: data)
      } catch {
        throw EdgeToolsFMConversionError.invalidGenerationSchema(
          description: String(describing: error)
        )
      }
    }
  }

  // MARK: - GenerationSchema

  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  extension GenerationSchema {
    public init(edgeToolsGenerationSchema: EdgeToolsGenerationSchema) throws {
      do {
        let schema = try edgeToolsGenerationSchema.foundationModelsNormalized()
        let data = try JSONEncoder().encode(schema)
        self = try JSONDecoder().decode(Self.self, from: data)
      } catch let error as EdgeToolsFMConversionError {
        throw error
      } catch {
        throw EdgeToolsFMConversionError.invalidGenerationSchema(
          description: String(describing: error)
        )
      }
    }
  }

  // MARK: - Normalization

  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  extension EdgeToolsGenerationSchema {
    fileprivate func foundationModelsNormalized(path: String = "$") throws -> Self {
      guard case .object(var object) = self else {
        throw EdgeToolsFMConversionError.unsupportedDynamicSchema(
          path: path,
          keyword: "boolean schema"
        )
      }

      try Self.normalizeType(in: &object, path: path)
      let propertyKeys = try Self.normalizeProperties(in: &object, path: path)
      try Self.normalizeSchemaValues(in: &object, path: path)
      try Self.normalizeSchemaArrays(in: &object, path: path)

      var schema = Self.object(object)
      if let propertyKeys {
        schema.merge(.xOrder(propertyKeys))
      }
      return schema
    }

    private static func normalizeType(
      in object: inout OrderedDictionary<Key, EdgeToolsValue>,
      path: String
    ) throws {
      guard case .array(let types)? = object[.type] else { return }
      guard object[.anyOf] == nil else {
        throw EdgeToolsFMConversionError.malformedSchema(
          path: path,
          description: "A schema cannot combine a type union with anyOf."
        )
      }

      let nonNullTypes = types.filter { $0 != .string("null") }
      if nonNullTypes.count == 1, nonNullTypes.count != types.count {
        object[.type] = nonNullTypes[0]
      } else {
        object[.type] = nil
        object[.anyOf] = .array(types.map { Self([.type: $0]).edgeToolsValue })
        object[.title] = object[.title] ?? .string(Self.inferredName(from: path))
      }
    }

    private static func normalizeProperties(
      in object: inout OrderedDictionary<Key, EdgeToolsValue>,
      path: String
    ) throws -> [String]? {
      guard case .object(let properties)? = object[.properties] else { return nil }
      object[.properties] = .object(
        try OrderedDictionary(
          uniqueKeysWithValues: properties.map { key, value in
            let propertyPath = "\(path).properties.\(key)"
            return (key, try Self.normalizedSchemaValue(value, path: propertyPath))
          }
        )
      )
      object[.additionalProperties] = object[.additionalProperties] ?? .boolean(false)
      return Array(properties.keys)
    }

    private static func normalizeSchemaValues(
      in object: inout OrderedDictionary<Key, EdgeToolsValue>,
      path: String
    ) throws {
      for key in [
        Key.items, .additionalProperties, .not, .if, .then, .else, .contains, .propertyNames
      ] {
        guard let value = object[key], case .object = value else { continue }
        object[key] = try Self.normalizedSchemaValue(value, path: "\(path).\(key.rawValue)")
      }
    }

    private static func normalizeSchemaArrays(
      in object: inout OrderedDictionary<Key, EdgeToolsValue>,
      path: String
    ) throws {
      for key in [Key.anyOf, .allOf, .oneOf, .prefixItems] {
        guard case .array(let values)? = object[key] else { continue }
        object[key] = .array(
          try values.enumerated()
            .map { index, value in
              try Self.normalizedSchemaValue(value, path: "\(path).\(key.rawValue)[\(index)]")
            }
        )
      }
    }

    private static func normalizedSchemaValue(
      _ value: EdgeToolsValue,
      path: String
    ) throws -> EdgeToolsValue {
      try Self.schema(from: value, path: path)
        .foundationModelsNormalized(path: path)
        .edgeToolsValue
    }

    static func schema(from value: EdgeToolsValue, path: String) throws -> Self {
      switch value {
      case .boolean(let value): .boolean(value)
      case .object(let object):
        .object(
          OrderedDictionary(
            uniqueKeysWithValues: object.map { (Key(rawValue: $0.key), $0.value) }
          )
        )
      default:
        throw EdgeToolsFMConversionError.malformedSchema(
          path: path,
          description: "Expected a schema object."
        )
      }
    }

    private static func inferredName(from path: String) -> String {
      let name = path.schemaPascalCased
      return name.isEmpty ? "GeneratedContent" : name
    }
  }

  extension String {
    var schemaPascalCased: String {
      self.split { !$0.isLetter && !$0.isNumber }
        .map { $0.prefix(1).uppercased() + $0.dropFirst() }
        .joined()
    }
  }

  // MARK: - Standard Keys

  extension EdgeToolsGenerationSchema {
    public static func xOrder(_ keys: some Sequence<String>) -> Self {
      Self.raw(.xOrder, .array(keys.map(EdgeToolsValue.string)))
    }
  }

  extension EdgeToolsGenerationSchema.Key {
    public static let xOrder: Self = "x-order"
  }
#endif
