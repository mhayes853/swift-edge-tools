#if FoundationModels && canImport(FoundationModels)
  import _EdgeToolsFoundation
  import FoundationModels
  import OrderedCollections
#endif

// MARK: - Foundation Models Conversion Errors

#if FoundationModels && canImport(FoundationModels)
  public struct EdgeToolsFMError: Error, Hashable, Sendable {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let nonFiniteNumber = Self(rawValue: "non-finite-number")
      public static let missingSchemaName = Self(rawValue: "missing-schema-name")
      public static let malformedSchema = Self(rawValue: "malformed-schema")
      public static let unsupportedDynamicSchema = Self(rawValue: "unsupported-dynamic-schema")
      public static let invalidGenerationSchema = Self(rawValue: "invalid-generation-schema")
    }

    public let code: Code
    public let message: String

    public init(code: Code, message: String) {
      self.code = code
      self.message = message
    }

    static func malformedSchema(path: String, description: String) -> Self {
      Self(code: .malformedSchema, message: "Malformed schema at \(path): \(description)")
    }

    static func unsupportedDynamicSchema(path: String, keyword: String) -> Self {
      Self(
        code: .unsupportedDynamicSchema,
        message: "DynamicGenerationSchema does not support '\(keyword)' at \(path)."
      )
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
        throw EdgeToolsFMError.malformedSchema(
          path: "$",
          description: "Unsupported generated content kind."
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
            throw EdgeToolsFMError(
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
        throw EdgeToolsFMError.invalidGenerationSchema(
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
      } catch let error as EdgeToolsFMError {
        throw error
      } catch {
        throw EdgeToolsFMError.invalidGenerationSchema(
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
        throw EdgeToolsFMError.unsupportedDynamicSchema(
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
        throw EdgeToolsFMError.malformedSchema(
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
        throw EdgeToolsFMError.malformedSchema(
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

// MARK: - Dynamic Schema Conversion

#if FoundationModels && canImport(FoundationModels)
  // MARK: - DynamicGenerationSchema

  @available(iOS 26.4, macOS 26.4, watchOS 27.0, tvOS 26.4, visionOS 26.4, *)
  extension DynamicGenerationSchema {
    public init(edgeToolsGenerationSchema: EdgeToolsGenerationSchema, name: String? = nil) throws {
      self = try FMSchemaConverter().convert(edgeToolsGenerationSchema, name: name, path: "$")
    }
  }

  // MARK: - Converter

  @available(iOS 26.4, macOS 26.4, watchOS 27.0, tvOS 26.4, visionOS 26.4, *)
  private struct FMSchemaConverter {
    func convert(
      _ schema: EdgeToolsGenerationSchema,
      name: String?,
      path: String
    ) throws -> DynamicGenerationSchema {
      let node = try FMSchemaNode(schema: schema, path: path)
      try node.validateSupportedKeywords()

      if let choices = try node.schemas(for: .anyOf) {
        return try self.union(
          choices,
          name: try node.name(overriding: name),
          description: node.description,
          path: "\(path).anyOf"
        )
      }

      let types = try node.types()
      guard types.count == 1, let type = types.first else {
        let choices = types.map { EdgeToolsGenerationSchema(.type($0)) }
        return try self.union(
          choices,
          name: try node.name(overriding: name),
          description: node.description,
          path: "\(path).type"
        )
      }
      return try self.convert(type, node: node, name: name)
    }

    private func convert(
      _ type: EdgeToolsGenerationSchema.ValueType,
      node: FMSchemaNode,
      name: String?
    ) throws -> DynamicGenerationSchema {
      switch type {
      case .string: try self.string(node: node, name: name)
      case .integer: try self.integer(node: node)
      case .number: try self.number(node: node)
      case .boolean: DynamicGenerationSchema(type: Bool.self)
      case .null: .null
      case .array: try self.array(node: node)
      case .object: try self.object(node: node, name: name)
      default:
        throw EdgeToolsFMError.malformedSchema(
          path: node.path,
          description: "Expected exactly one supported JSON schema type."
        )
      }
    }

    private func union(
      _ choices: [EdgeToolsGenerationSchema],
      name: String,
      description: String?,
      path: String
    ) throws -> DynamicGenerationSchema {
      guard !choices.isEmpty else {
        throw EdgeToolsFMError.malformedSchema(
          path: path,
          description: "A dynamic schema must declare a type or anyOf choices."
        )
      }
      return DynamicGenerationSchema(
        name: name,
        description: description,
        anyOf: try choices.enumerated()
          .map { index, choice in
            try self.convert(choice, name: "\(name)Choice\(index + 1)", path: "\(path)[\(index)]")
          }
      )
    }
  }

  @available(iOS 26.4, macOS 26.4, watchOS 27.0, tvOS 26.4, visionOS 26.4, *)
  extension FMSchemaConverter {
    private func string(node: FMSchemaNode, name: String?) throws -> DynamicGenerationSchema {
      if let values = try node.strings(for: .enum) {
        return DynamicGenerationSchema(
          name: try node.name(overriding: name),
          description: node.description,
          anyOf: values
        )
      }

      let constant = node.string(for: .const).map(GenerationGuide<String>.constant)
      let pattern = try node.string(for: .pattern)
        .map { pattern in
          do {
            return GenerationGuide<String>.pattern(try Regex(pattern))
          } catch {
            throw EdgeToolsFMError.malformedSchema(
              path: "\(node.path).pattern",
              description: String(describing: error)
            )
          }
        }
      return DynamicGenerationSchema(
        type: String.self,
        guides: [constant, pattern].compactMap { $0 }
      )
    }

    private func integer(node: FMSchemaNode) throws -> DynamicGenerationSchema {
      let minimum = try node.int(for: .minimum).map(GenerationGuide<Int>.minimum)
      let maximum = try node.int(for: .maximum).map(GenerationGuide<Int>.maximum)
      return DynamicGenerationSchema(type: Int.self, guides: [minimum, maximum].compactMap { $0 })
    }

    private func number(node: FMSchemaNode) throws -> DynamicGenerationSchema {
      let minimum = try node.double(for: .minimum).map(GenerationGuide<Double>.minimum)
      let maximum = try node.double(for: .maximum).map(GenerationGuide<Double>.maximum)
      return DynamicGenerationSchema(
        type: Double.self,
        guides: [minimum, maximum].compactMap { $0 }
      )
    }

    private func array(node: FMSchemaNode) throws -> DynamicGenerationSchema {
      let itemSchema = try node.schema(for: .items)
      return DynamicGenerationSchema(
        arrayOf: try self.convert(itemSchema, name: nil, path: "\(node.path).items"),
        minimumElements: try node.int(for: .minItems),
        maximumElements: try node.int(for: .maxItems)
      )
    }

    private func object(node: FMSchemaNode, name: String?) throws -> DynamicGenerationSchema {
      try node.validateAdditionalProperties()
      let required = try node.requiredKeys()
      let properties = try node.properties()
        .map { key, schema in
          let propertyNode = try FMSchemaNode(
            schema: schema,
            path: "\(node.path).properties.\(key)"
          )
          return DynamicGenerationSchema.Property(
            name: key,
            description: propertyNode.description,
            schema: try self.convert(
              schema,
              name: key.schemaPascalCased,
              path: propertyNode.path
            ),
            isOptional: !required.contains(key)
          )
        }
      return DynamicGenerationSchema(
        name: try node.name(overriding: name),
        description: node.description,
        properties: properties
      )
    }

  }

  // MARK: - Schema Node

  @available(iOS 26.4, macOS 26.4, watchOS 27.0, tvOS 26.4, visionOS 26.4, *)
  private struct FMSchemaNode {
    typealias Object = OrderedDictionary<EdgeToolsGenerationSchema.Key, EdgeToolsValue>

    let object: Object
    let path: String

    init(schema: EdgeToolsGenerationSchema, path: String) throws {
      guard case .object(let object) = schema else {
        throw EdgeToolsFMError.unsupportedDynamicSchema(
          path: path,
          keyword: "boolean schema"
        )
      }
      self.object = object
      self.path = path
    }

    var description: String? {
      self.string(for: .description)
    }

    func name(overriding name: String?) throws -> String {
      guard let name = name ?? self.string(for: .title) else {
        throw EdgeToolsFMError(
          code: .missingSchemaName,
          message: "A FoundationModels schema at \(self.path) requires a name."
        )
      }
      return name
    }

    func types() throws -> [EdgeToolsGenerationSchema.ValueType] {
      guard let value = self.object[.type] else { return [] }
      guard let types = EdgeToolsGenerationSchema.valueType(from: value) else {
        throw EdgeToolsFMError.malformedSchema(
          path: "\(self.path).type",
          description: "Invalid JSON schema type."
        )
      }
      return types.orderedMembers
    }

    func schema(for key: EdgeToolsGenerationSchema.Key) throws -> EdgeToolsGenerationSchema {
      guard let value = self.object[key] else {
        throw EdgeToolsFMError.malformedSchema(
          path: "\(self.path).\(key.rawValue)",
          description: "Expected a schema."
        )
      }
      return try EdgeToolsGenerationSchema.schema(
        from: value,
        path: "\(self.path).\(key.rawValue)"
      )
    }

    func schemas(
      for key: EdgeToolsGenerationSchema.Key
    ) throws -> [EdgeToolsGenerationSchema]? {
      guard let value = self.object[key] else { return nil }
      guard case .array(let values) = value else {
        throw EdgeToolsFMError.malformedSchema(
          path: "\(self.path).\(key.rawValue)",
          description: "Expected an array of schemas."
        )
      }
      return try values.enumerated()
        .map { index, value in
          try EdgeToolsGenerationSchema.schema(
            from: value,
            path: "\(self.path).\(key.rawValue)[\(index)]"
          )
        }
    }

    func properties() throws -> OrderedDictionary<String, EdgeToolsGenerationSchema> {
      guard let value = self.object[.properties] else { return [:] }
      guard case .object(let properties) = value else {
        throw EdgeToolsFMError.malformedSchema(
          path: "\(self.path).properties",
          description: "Expected an object."
        )
      }
      return try OrderedDictionary(
        uniqueKeysWithValues: properties.map { key, value in
          (
            key,
            try EdgeToolsGenerationSchema.schema(
              from: value,
              path: "\(self.path).properties.\(key)"
            )
          )
        }
      )
    }

    func requiredKeys() throws -> Set<String> {
      guard let value = self.object[.required] else { return [] }
      guard case .array(let values) = value else {
        throw EdgeToolsFMError.malformedSchema(
          path: "\(self.path).required",
          description: "Expected an array of property names."
        )
      }
      return try Set(
        values.map { value in
          guard case .string(let key) = value else {
            throw EdgeToolsFMError.malformedSchema(
              path: "\(self.path).required",
              description: "Expected an array of property names."
            )
          }
          return key
        }
      )
    }

    func strings(for key: EdgeToolsGenerationSchema.Key) throws -> [String]? {
      guard let value = self.object[key] else { return nil }
      guard case .array(let values) = value else {
        throw EdgeToolsFMError.malformedSchema(
          path: "\(self.path).\(key.rawValue)",
          description: "Expected an array of strings."
        )
      }
      return try values.map { value in
        guard case .string(let string) = value else {
          throw EdgeToolsFMError.unsupportedDynamicSchema(
            path: "\(self.path).\(key.rawValue)",
            keyword: key.rawValue
          )
        }
        return string
      }
    }

    func string(for key: EdgeToolsGenerationSchema.Key) -> String? {
      guard case .string(let value)? = self.object[key] else { return nil }
      return value
    }

    func int(for key: EdgeToolsGenerationSchema.Key) throws -> Int? {
      guard let value = self.object[key] else { return nil }
      return switch value {
      case .integer(let value): value
      case .number(let value):
        try Int(exactly: value).fmUnwrapped(path: "\(self.path).\(key.rawValue)", type: "integer")
      default:
        throw EdgeToolsFMError.malformedSchema(
          path: "\(self.path).\(key.rawValue)",
          description: "Expected an integer."
        )
      }
    }

    func double(for key: EdgeToolsGenerationSchema.Key) throws -> Double? {
      guard let value = self.object[key] else { return nil }
      return switch value {
      case .integer(let value): Double(value)
      case .number(let value): value
      default:
        throw EdgeToolsFMError.malformedSchema(
          path: "\(self.path).\(key.rawValue)",
          description: "Expected a number."
        )
      }
    }

    func validateAdditionalProperties() throws {
      guard let value = self.object[.additionalProperties], value != .boolean(false) else { return }
      throw EdgeToolsFMError.unsupportedDynamicSchema(
        path: "\(self.path).additionalProperties",
        keyword: "additionalProperties"
      )
    }

    func validateSupportedKeywords() throws {
      let unsupported: [EdgeToolsGenerationSchema.Key] = [
        .allOf, .oneOf, .not, .if, .then, .else, .prefixItems, .contains,
        .patternProperties, .propertyNames, .minProperties, .maxProperties,
        .dependentRequired, .uniqueItems, .minContains, .maxContains, .minLength,
        .maxLength, .multipleOf, .exclusiveMinimum, .exclusiveMaximum
      ]
      if let key = unsupported.first(where: { self.object[$0] != nil }) {
        throw EdgeToolsFMError.unsupportedDynamicSchema(
          path: self.path,
          keyword: key.rawValue
        )
      }
    }

  }

  // MARK: - Helpers

  extension Optional {
    fileprivate func fmUnwrapped(path: String, type: String) throws -> Wrapped {
      guard let value = self else {
        throw EdgeToolsFMError.malformedSchema(
          path: path,
          description: "Expected an \(type)."
        )
      }
      return value
    }
  }

  extension EdgeToolsGenerationSchema.ValueType {
    fileprivate var orderedMembers: [Self] {
      [.string, .integer, .number, .boolean, .null, .array, .object].filter(self.contains)
    }
  }
#endif
