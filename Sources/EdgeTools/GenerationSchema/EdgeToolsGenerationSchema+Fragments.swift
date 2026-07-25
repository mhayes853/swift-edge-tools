import OrderedCollections

// MARK: - Fragment helpers

extension EdgeToolsGenerationSchema {
  public static var string: Self { .type(.string) }
  public static var number: Self { .type(.number) }
  public static var integer: Self { .type(.integer) }
  public static var null: Self { .type(.null) }
  public static var boolean: Self { .type(.boolean) }

  /// The universal generation schema, equivalent to JSON Schema `true` — it
  /// accepts any well-formed value.
  public static var universal: Self { .boolean(true) }

  public static func type(_ type: ValueType) -> Self {
    Self([.type: type.edgeToolsValue])
  }

  public static func title(_ value: String) -> Self {
    Self([.title: .string(value)])
  }

  public static func description(_ value: String) -> Self {
    Self([.description: .string(value)])
  }

  public static func `default`(_ value: EdgeToolsValue) -> Self {
    Self([.default: value])
  }

  public static func examples(_ value: [EdgeToolsValue]) -> Self {
    Self([.examples: .array(value)])
  }

  public static func `enum`(_ value: [EdgeToolsValue]) -> Self {
    Self([.enum: .array(value)])
  }

  public static func const(_ value: EdgeToolsValue) -> Self {
    Self([.const: value])
  }

  public static func allOf(_ schemas: [Self]) -> Self {
    Self([.allOf: Self.schemaArrayValue(schemas)])
  }

  public static func anyOf(_ schemas: [Self]) -> Self {
    Self([.anyOf: Self.schemaArrayValue(schemas)])
  }

  public static func oneOf(_ schemas: [Self]) -> Self {
    Self([.oneOf: Self.schemaArrayValue(schemas)])
  }

  public static func not(_ schema: Self) -> Self {
    Self([.not: schema.edgeToolsValue])
  }

  public static func `if`(_ schema: Self) -> Self {
    Self([.if: schema.edgeToolsValue])
  }

  public static func then(_ schema: Self) -> Self {
    Self([.then: schema.edgeToolsValue])
  }

  public static func `else`(_ schema: Self) -> Self {
    Self([.else: schema.edgeToolsValue])
  }

  public static func properties(_ properties: KeyValuePairs<String, Self>) -> Self {
    Self([.properties: Self.schemaObjectValue(properties)])
  }

  public static func properties(_ properties: KeyValuePairs<String, [Self]>) -> Self {
    Self(
      [
        .properties: .object(
          OrderedDictionary(
            uniqueKeysWithValues: properties.map { entry in
              (entry.0, EdgeToolsGenerationSchema(entry.1).edgeToolsValue)
            }
          )
        )
      ]
    )
  }

  public static func property(_ key: String, _ schema: Self) -> Self {
    Self.properties(KeyValuePairs(dictionaryLiteral: (key, schema)))
  }

  public static func property(_ key: String, _ schemas: Self...) -> Self {
    Self.properties(KeyValuePairs(dictionaryLiteral: (key, Self(schemas))))
  }

  public static func required(_ keys: [String]) -> Self {
    Self([.required: Self.rawStringArray(keys)])
  }

  public static func additionalProperties(_ schema: Self) -> Self {
    Self([.additionalProperties: schema.edgeToolsValue])
  }

  public static func additionalProperties(_ allowed: Bool) -> Self {
    Self([.additionalProperties: .boolean(allowed)])
  }

  public static func patternProperties(_ properties: KeyValuePairs<String, Self>) -> Self {
    Self([.patternProperties: Self.schemaObjectValue(properties)])
  }

  public static func propertyNames(_ schema: Self) -> Self {
    Self([.propertyNames: schema.edgeToolsValue])
  }

  public static func minProperties(_ value: Int) -> Self {
    Self([.minProperties: .integer(value)])
  }

  public static func maxProperties(_ value: Int) -> Self {
    Self([.maxProperties: .integer(value)])
  }

  public static func dependentRequired(_ value: KeyValuePairs<String, [String]>) -> Self {
    Self(
      [
        .dependentRequired: Self.rawStringObject(
          OrderedDictionary(uniqueKeysWithValues: value.map { ($0.0, $0.1) })
        )
      ]
    )
  }

  public static func items(_ schema: Self) -> Self {
    Self([.items: schema.edgeToolsValue])
  }

  public static func prefixItems(_ schemas: [Self]) -> Self {
    Self([.prefixItems: Self.schemaArrayValue(schemas)])
  }

  public static func minItems(_ value: Int) -> Self {
    Self([.minItems: .integer(value)])
  }

  public static func maxItems(_ value: Int) -> Self {
    Self([.maxItems: .integer(value)])
  }

  public static func uniqueItems(_ value: Bool = true) -> Self {
    Self([.uniqueItems: .boolean(value)])
  }

  public static func contains(_ schema: Self) -> Self {
    Self([.contains: schema.edgeToolsValue])
  }

  public static func minContains(_ value: Int) -> Self {
    Self([.minContains: .integer(value)])
  }

  public static func maxContains(_ value: Int) -> Self {
    Self([.maxContains: .integer(value)])
  }

  public static func minLength(_ value: Int) -> Self {
    Self([.minLength: .integer(value)])
  }

  public static func maxLength(_ value: Int) -> Self {
    Self([.maxLength: .integer(value)])
  }

  public static func pattern(_ value: String) -> Self {
    Self([.pattern: .string(value)])
  }

  public static func lengthRange(_ range: ClosedRange<Int>) -> Self {
    Self(.minLength(range.lowerBound), .maxLength(range.upperBound))
  }

  public static func lengthRange(_ range: PartialRangeFrom<Int>) -> Self {
    Self(.minLength(range.lowerBound))
  }

  public static func lengthRange(_ range: PartialRangeThrough<Int>) -> Self {
    Self(.maxLength(range.upperBound))
  }

  public static func lengthRange(_ range: Range<Int>) -> Self {
    Self(.minLength(range.lowerBound), .maxLength(range.upperBound - 1))
  }

  public static func multipleOf(_ value: Int) -> Self {
    Self([.multipleOf: .integer(value)])
  }

  public static func multipleOf(_ value: Double) -> Self {
    Self([.multipleOf: .number(value)])
  }

  public static func minimum(_ value: Int) -> Self {
    Self([.minimum: .integer(value)])
  }

  public static func minimum(_ value: Double) -> Self {
    Self([.minimum: .number(value)])
  }

  public static func exclusiveMinimum(_ value: Int) -> Self {
    Self([.exclusiveMinimum: .integer(value)])
  }

  public static func exclusiveMinimum(_ value: Double) -> Self {
    Self([.exclusiveMinimum: .number(value)])
  }

  public static func maximum(_ value: Int) -> Self {
    Self([.maximum: .integer(value)])
  }

  public static func maximum(_ value: Double) -> Self {
    Self([.maximum: .number(value)])
  }

  public static func exclusiveMaximum(_ value: Int) -> Self {
    Self([.exclusiveMaximum: .integer(value)])
  }

  public static func exclusiveMaximum(_ value: Double) -> Self {
    Self([.exclusiveMaximum: .number(value)])
  }

  public static func range(_ range: ClosedRange<Int>) -> Self {
    Self(.minimum(range.lowerBound), .maximum(range.upperBound))
  }

  public static func range(_ range: ClosedRange<Double>) -> Self {
    Self(.minimum(range.lowerBound), .maximum(range.upperBound))
  }

  public static func range(_ range: PartialRangeFrom<Int>) -> Self {
    Self(.minimum(range.lowerBound))
  }

  public static func range(_ range: PartialRangeFrom<Double>) -> Self {
    Self(.minimum(range.lowerBound))
  }

  public static func range(_ range: PartialRangeThrough<Int>) -> Self {
    Self(.maximum(range.upperBound))
  }

  public static func range(_ range: PartialRangeThrough<Double>) -> Self {
    Self(.maximum(range.upperBound))
  }

  public static func range(_ range: Range<Int>) -> Self {
    Self(.minimum(range.lowerBound), .exclusiveMaximum(range.upperBound))
  }

  public static func range(_ range: Range<Double>) -> Self {
    Self(.minimum(range.lowerBound), .exclusiveMaximum(range.upperBound))
  }
}

