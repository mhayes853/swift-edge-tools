#if FoundationModels && canImport(FoundationModels)
  import FoundationModels
  import OrderedCollections

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
        throw EdgeToolsFMConversionError.malformedSchema(
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
        throw EdgeToolsFMConversionError.malformedSchema(
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
            throw EdgeToolsFMConversionError.malformedSchema(
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
            schema: try self.convert(schema, name: Self.name(from: key), path: propertyNode.path),
            isOptional: !required.contains(key)
          )
        }
      return DynamicGenerationSchema(
        name: try node.name(overriding: name),
        description: node.description,
        properties: properties
      )
    }

    private static func name(from value: String) -> String {
      let components = value.split { !$0.isLetter && !$0.isNumber }
      return components.map { $0.prefix(1).uppercased() + $0.dropFirst() }.joined()
    }
  }

  // MARK: - Schema Node

  private struct FMSchemaNode {
    typealias Object = OrderedDictionary<EdgeToolsGenerationSchema.Key, EdgeToolsValue>

    let object: Object
    let path: String

    init(schema: EdgeToolsGenerationSchema, path: String) throws {
      guard case .object(let object) = schema else {
        throw EdgeToolsFMConversionError.unsupportedDynamicSchema(
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
        throw EdgeToolsFMConversionError.missingSchemaName(path: self.path)
      }
      return name
    }

    func types() throws -> [EdgeToolsGenerationSchema.ValueType] {
      guard let value = self.object[.type] else { return [] }
      guard let types = EdgeToolsGenerationSchema.valueType(from: value) else {
        throw EdgeToolsFMConversionError.malformedSchema(
          path: "\(self.path).type",
          description: "Invalid JSON schema type."
        )
      }
      return types.orderedMembers
    }

    func schema(for key: EdgeToolsGenerationSchema.Key) throws -> EdgeToolsGenerationSchema {
      guard let value = self.object[key] else {
        throw EdgeToolsFMConversionError.malformedSchema(
          path: "\(self.path).\(key.rawValue)",
          description: "Expected a schema."
        )
      }
      return try Self.schema(from: value, path: "\(self.path).\(key.rawValue)")
    }

    func schemas(
      for key: EdgeToolsGenerationSchema.Key
    ) throws -> [EdgeToolsGenerationSchema]? {
      guard let value = self.object[key] else { return nil }
      guard case .array(let values) = value else {
        throw EdgeToolsFMConversionError.malformedSchema(
          path: "\(self.path).\(key.rawValue)",
          description: "Expected an array of schemas."
        )
      }
      return try values.enumerated()
        .map { index, value in
          try Self.schema(from: value, path: "\(self.path).\(key.rawValue)[\(index)]")
        }
    }

    func properties() throws -> OrderedDictionary<String, EdgeToolsGenerationSchema> {
      guard let value = self.object[.properties] else { return [:] }
      guard case .object(let properties) = value else {
        throw EdgeToolsFMConversionError.malformedSchema(
          path: "\(self.path).properties",
          description: "Expected an object."
        )
      }
      return try OrderedDictionary(
        uniqueKeysWithValues: properties.map { key, value in
          (key, try Self.schema(from: value, path: "\(self.path).properties.\(key)"))
        }
      )
    }

    func requiredKeys() throws -> Set<String> {
      guard let value = self.object[.required] else { return [] }
      guard case .array(let values) = value else {
        throw EdgeToolsFMConversionError.malformedSchema(
          path: "\(self.path).required",
          description: "Expected an array of property names."
        )
      }
      return try Set(
        values.map { value in
          guard case .string(let key) = value else {
            throw EdgeToolsFMConversionError.malformedSchema(
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
        throw EdgeToolsFMConversionError.malformedSchema(
          path: "\(self.path).\(key.rawValue)",
          description: "Expected an array of strings."
        )
      }
      return try values.map { value in
        guard case .string(let string) = value else {
          throw EdgeToolsFMConversionError.unsupportedDynamicSchema(
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
        throw EdgeToolsFMConversionError.malformedSchema(
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
        throw EdgeToolsFMConversionError.malformedSchema(
          path: "\(self.path).\(key.rawValue)",
          description: "Expected a number."
        )
      }
    }

    func validateAdditionalProperties() throws {
      guard let value = self.object[.additionalProperties], value != .boolean(false) else { return }
      throw EdgeToolsFMConversionError.unsupportedDynamicSchema(
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
        throw EdgeToolsFMConversionError.unsupportedDynamicSchema(
          path: self.path,
          keyword: key.rawValue
        )
      }
    }

    private static func schema(
      from value: EdgeToolsValue,
      path: String
    ) throws -> EdgeToolsGenerationSchema {
      switch value {
      case .boolean(let value): .boolean(value)
      case .object(let object):
        .object(
          OrderedDictionary(
            uniqueKeysWithValues: object.map {
              (EdgeToolsGenerationSchema.Key(rawValue: $0.key), $0.value)
            }
          )
        )
      default:
        throw EdgeToolsFMConversionError.malformedSchema(
          path: path,
          description: "Expected a schema object."
        )
      }
    }
  }

  // MARK: - Helpers

  extension Optional {
    fileprivate func fmUnwrapped(path: String, type: String) throws -> Wrapped {
      guard let value = self else {
        throw EdgeToolsFMConversionError.malformedSchema(
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
