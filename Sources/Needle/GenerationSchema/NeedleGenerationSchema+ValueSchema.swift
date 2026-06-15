// MARK: - ValueSchema

extension NeedleGenerationSchema {
  /// The type of value represented by a ``NeedleGenerationSchema``.
  public struct ValueSchema: Hashable, Sendable, Codable {
    /// A string type.
    public var string: String?

    /// Whether or not the type indicates that the value can be a boolean.
    public var isBoolean: Bool

    /// An array type.
    public var array: Array?

    /// An object type.
    public var object: Object?

    /// A number type.
    ///
    /// If this value is present with ``integer``, then the properties from `number` will override
    /// the integer properties when encoding.
    public var number: Number?

    /// An integer type.
    ///
    /// If this value is present with ``number``, then the properties from `number` will override
    /// the integer properties when encoding.
    public var integer: Integer?

    /// Whether or not the type is nullable.
    public var isNullable: Bool

    /// A union type.
    ///
    /// - Parameters:
    ///   - string: A string type.
    ///   - isBoolean: Whether or not the type indicates that the value can be a boolean.
    ///   - array: An array type.
    ///   - object: An object type.
    ///   - number: A number type.
    ///   - integer: An integer type.
    ///   - isNullable: Whether or not the type is nullable.
    public static func union(
      string: String? = nil,
      isBoolean: Bool = false,
      array: Array? = nil,
      object: Object? = nil,
      number: Number? = nil,
      integer: Integer? = nil,
      isNullable: Bool = false
    ) -> Self {
      Self(
        string: string,
        isBoolean: isBoolean,
        array: array,
        object: object,
        number: number,
        integer: integer,
        isNullable: isNullable
      )
    }

    /// A nullable type.
    public static let null = Self.union(isNullable: true)

    /// A boolean type.
    public static let boolean = Self.union(isBoolean: true)
  }
}

// MARK: - ValueTypes

extension NeedleGenerationSchema.ValueSchema {
  /// All of the ``NeedleGenerationSchema/ValueType`` instances that this value schema represents.
  public var type: NeedleGenerationSchema.ValueType {
    var schemaTypes = NeedleGenerationSchema.ValueType()
    if self.array != nil {
      schemaTypes.insert(.array)
    }
    if self.integer != nil {
      schemaTypes.insert(.integer)
    }
    if self.number != nil {
      schemaTypes.insert(.number)
    }
    if self.string != nil {
      schemaTypes.insert(.string)
    }
    if self.object != nil {
      schemaTypes.insert(.object)
    }
    if self.isBoolean {
      schemaTypes.insert(.boolean)
    }
    if self.isNullable {
      schemaTypes.insert(.null)
    }
    return schemaTypes
  }
}

// MARK: - String

extension NeedleGenerationSchema.ValueSchema {
  /// A string-specific schema.
  public struct String: Hashable, Sendable, Codable {
    /// The minimum length of the string.
    ///
    /// [minLength](https://json-schema.org/draft/2020-12/json-schema-validation#name-minlength)
    public var minLength: Int?

    /// The maximum length of the string.
    ///
    /// [maxLength](https://json-schema.org/draft/2020-12/json-schema-validation#name-maxlength)
    public var maxLength: Int?

    /// A regular expression that the string must match.
    ///
    /// [pattern](https://json-schema.org/draft/2020-12/json-schema-validation#name-pattern)
    public var pattern: Swift.String?

    /// Creates a string-specific schema.
    ///
    /// - Parameters:
    ///   - minLength: The minimum length of the string.
    ///   - maxLength: The maximum length of the string.
    ///   - pattern: A regular expression that the string must match.
    public static func string(
      minLength: Int? = nil,
      maxLength: Int? = nil,
      pattern: Swift.String? = nil
    ) -> Self {
      Self(minLength: minLength, maxLength: maxLength, pattern: pattern)
    }
  }

  /// Creates a string-specific schema.
  ///
  /// - Parameters:
  ///   - minLength: The minimum length of the string.
  ///   - maxLength: The maximum length of the string.
  ///   - pattern: A regular expression that the string must match.
  public static func string(
    minLength: Int? = nil,
    maxLength: Int? = nil,
    pattern: Swift.String? = nil
  ) -> Self {
    .union(string: .string(minLength: minLength, maxLength: maxLength, pattern: pattern))
  }
}

// MARK: - Number

extension NeedleGenerationSchema.ValueSchema {
  /// A number-specific schema.
  public struct Number: Hashable, Sendable, Codable {
    /// The value that the number must be a multiple of.
    ///
    /// [multipleOf](https://json-schema.org/draft/2020-12/json-schema-validation#name-multipleof)
    public var multipleOf: Double?

    /// The minimum value (inclusive) of the number.
    ///
    /// [minimum](https://json-schema.org/draft/2020-12/json-schema-validation#name-minimum)
    public var minimum: Double?

    /// The minimum value (exclusive) of the number.
    ///
    /// [exclusiveMinimum](https://json-schema.org/draft/2020-12/json-schema-validation#name-exclusiveminimum)
    public var exclusiveMinimum: Double?

    /// The maximum value (inclusive) of the number.
    ///
    /// [maximum](https://json-schema.org/draft/2020-12/json-schema-validation#name-maximum)
    public var maximum: Double?

    /// The maximum value (exclusive) of the number.
    ///
    /// [exclusiveMaximum](https://json-schema.org/draft/2020-12/json-schema-validation#name-exclusivemaximum)
    public var exclusiveMaximum: Double?

    /// Creates a number-specific schema.
    ///
    /// - Parameters:
    ///   - multipleOf: The value that the number must be a multiple of.
    ///   - minimum: The minimum value (inclusive) of the number.
    ///   - exclusiveMinimum: The minimum value (exclusive) of the number.
    ///   - maximum: The maximum value (inclusive) of the number.
    ///   - exclusiveMaximum: The maximum value (exclusive) of the number.
    public static func number(
      multipleOf: Double? = nil,
      minimum: Double? = nil,
      exclusiveMinimum: Double? = nil,
      maximum: Double? = nil,
      exclusiveMaximum: Double? = nil
    ) -> Self {
      Self(
        multipleOf: multipleOf,
        minimum: minimum,
        exclusiveMinimum: exclusiveMinimum,
        maximum: maximum,
        exclusiveMaximum: exclusiveMaximum
      )
    }
  }

  /// Creates a number-specific schema.
  ///
  /// - Parameters:
  ///   - multipleOf: The value that the number must be a multiple of.
  ///   - minimum: The minimum value (inclusive) of the number.
  ///   - exclusiveMinimum: The minimum value (exclusive) of the number.
  ///   - maximum: The maximum value (inclusive) of the number.
  ///   - exclusiveMaximum: The maximum value (exclusive) of the number.
  public static func number(
    multipleOf: Double? = nil,
    minimum: Double? = nil,
    exclusiveMinimum: Double? = nil,
    maximum: Double? = nil,
    exclusiveMaximum: Double? = nil
  ) -> Self {
    .union(
      number: .number(
        multipleOf: multipleOf,
        minimum: minimum,
        exclusiveMinimum: exclusiveMinimum,
        maximum: maximum,
        exclusiveMaximum: exclusiveMaximum
      )
    )
  }
}

// MARK: - Integer

extension NeedleGenerationSchema.ValueSchema {
  /// An integer-specific schema.
  public struct Integer: Hashable, Sendable, Codable {
    /// The value that the integer must be a multiple of.
    ///
    /// [multipleOf](https://json-schema.org/draft/2020-12/json-schema-validation#name-multipleof)
    public var multipleOf: Int?

    /// The minimum value (inclusive) of the integer.
    ///
    /// [minimum](https://json-schema.org/draft/2020-12/json-schema-validation#name-minimum)
    public var minimum: Int?

    /// The minimum value (exclusive) of the integer.
    ///
    /// [exclusiveMinimum](https://json-schema.org/draft/2020-12/json-schema-validation#name-exclusiveminimum)
    public var exclusiveMinimum: Int?

    /// The maximum value (inclusive) of the integer.
    ///
    /// [maximum](https://json-schema.org/draft/2020-12/json-schema-validation#name-maximum)
    public var maximum: Int?

    /// The maximum value (exclusive) of the integer.
    ///
    /// [exclusiveMaximum](https://json-schema.org/draft/2020-12/json-schema-validation#name-exclusivemaximum)
    public var exclusiveMaximum: Int?

    /// Creates a number-specific schema.
    ///
    /// - Parameters:
    ///   - multipleOf: The value that the integer must be a multiple of.
    ///   - minimum: The minimum value (inclusive) of the integer.
    ///   - exclusiveMinimum: The minimum value (exclusive) of the integer.
    ///   - maximum: The maximum value (inclusive) of the integer.
    ///   - exclusiveMaximum: The maximum value (exclusive) of the integer.
    public static func integer(
      multipleOf: Int? = nil,
      minimum: Int? = nil,
      exclusiveMinimum: Int? = nil,
      maximum: Int? = nil,
      exclusiveMaximum: Int? = nil
    ) -> Self {
      Self(
        multipleOf: multipleOf,
        minimum: minimum,
        exclusiveMinimum: exclusiveMinimum,
        maximum: maximum,
        exclusiveMaximum: exclusiveMaximum
      )
    }
  }

  /// Creates an integer-specific schema.
  ///
  /// - Parameters:
  ///   - multipleOf: The value that the integer must be a multiple of.
  ///   - minimum: The minimum value (inclusive) of the integer.
  ///   - exclusiveMinimum: The minimum value (exclusive) of the integer.
  ///   - maximum: The maximum value (inclusive) of the integer.
  ///   - exclusiveMaximum: The maximum value (exclusive) of the integer.
  public static func integer(
    multipleOf: Int? = nil,
    minimum: Int? = nil,
    exclusiveMinimum: Int? = nil,
    maximum: Int? = nil,
    exclusiveMaximum: Int? = nil
  ) -> Self {
    .union(
      integer: .integer(
        multipleOf: multipleOf,
        minimum: minimum,
        exclusiveMinimum: exclusiveMinimum,
        maximum: maximum,
        exclusiveMaximum: exclusiveMaximum
      )
    )
  }
}

// MARK: - Array

extension NeedleGenerationSchema.ValueSchema {
  /// An array-specific schema.
  public struct Array: Hashable, Sendable, Codable {
    /// The schema applied to every element in the array.
    ///
    /// [items](https://json-schema.org/draft/2020-12/json-schema-validation#name-items)
    public var items: NeedleGenerationSchema?

    /// An array of schemas, each applied to the element at the matching index.
    ///
    /// [prefixItems](https://json-schema.org/draft/2020-12/json-schema-validation#name-prefixitems)
    public var prefixItems: [NeedleGenerationSchema]?

    /// The minimum number of items allowed in the array.
    ///
    /// [minItems](https://json-schema.org/draft/2020-12/json-schema-validation#name-minitems)
    public var minItems: Int?

    /// The maximum number of items allowed in the array.
    ///
    /// [maxItems](https://json-schema.org/draft/2020-12/json-schema-validation#name-maxitems)
    public var maxItems: Int?

    /// A boolean that indicates whether all items in the array must be unique.
    ///
    /// [uniqueItems](https://json-schema.org/draft/2020-12/json-schema-validation#name-uniqueitems)
    public var uniqueItems: Bool?

    /// A schema that must be contained within the array.
    ///
    /// [contains](https://json-schema.org/draft/2020-12/json-schema-validation#name-contains)
    public var contains: NeedleGenerationSchema?

    /// The minimum number of array elements that must match ``contains``.
    ///
    /// Has no effect when ``contains`` is not present.
    ///
    /// [minContains](https://json-schema.org/draft/2020-12/json-schema-validation#name-mincontains)
    public var minContains: Int?

    /// The maximum number of array elements that may match ``contains``.
    ///
    /// Has no effect when ``contains`` is not present.
    ///
    /// [maxContains](https://json-schema.org/draft/2020-12/json-schema-validation#name-maxcontains)
    public var maxContains: Int?

    /// Creates an array-specific schema.
    ///
    /// - Parameters:
    ///   - items: The schema applied to every element in the array.
    ///   - prefixItems: An array of schemas applied to the element at the matching index.
    ///   - minItems: The minimum number of items allowed in the array.
    ///   - maxItems: The maximum number of items allowed in the array.
    ///   - uniqueItems: A boolean that indicates whether all items in the array must be unique.
    ///   - contains: A schema that must be contained within the array.
    ///   - minContains: The minimum number of array elements that must match ``contains``.
    ///   - maxContains: The maximum number of array elements that may match ``contains``.
    public static func array(
      items: NeedleGenerationSchema? = nil,
      prefixItems: [NeedleGenerationSchema]? = nil,
      minItems: Int? = nil,
      maxItems: Int? = nil,
      uniqueItems: Bool? = nil,
      contains: NeedleGenerationSchema? = nil,
      minContains: Int? = nil,
      maxContains: Int? = nil
    ) -> Self {
      Self(
        items: items,
        prefixItems: prefixItems,
        minItems: minItems,
        maxItems: maxItems,
        uniqueItems: uniqueItems,
        contains: contains,
        minContains: minContains,
        maxContains: maxContains
      )
    }
  }

  /// Creates an array-specific schema.
  ///
  /// - Parameters:
  ///   - items: The schema applied to every element in the array.
  ///   - prefixItems: An array of schemas applied to the element at the matching index.
  ///   - minItems: The minimum number of items allowed in the array.
  ///   - maxItems: The maximum number of items allowed in the array.
  ///   - uniqueItems: A boolean that indicates whether all items in the array must be unique.
  ///   - contains: A schema that must be contained within the array.
  ///   - minContains: The minimum number of array elements that must match `contains`.
  ///   - maxContains: The maximum number of array elements that may match `contains`.
  public static func array(
    items: NeedleGenerationSchema? = nil,
    prefixItems: [NeedleGenerationSchema]? = nil,
    minItems: Int? = nil,
    maxItems: Int? = nil,
    uniqueItems: Bool? = nil,
    contains: NeedleGenerationSchema? = nil,
    minContains: Int? = nil,
    maxContains: Int? = nil
  ) -> Self {
    .union(
      array: .array(
        items: items,
        prefixItems: prefixItems,
        minItems: minItems,
        maxItems: maxItems,
        uniqueItems: uniqueItems,
        contains: contains,
        minContains: minContains,
        maxContains: maxContains
      )
    )
  }
}

// MARK: - Object

extension NeedleGenerationSchema.ValueSchema {
  /// An object-specific schema.
  ///
  /// [objects](https://json-schema.org/draft/2020-12/json-schema-validation#name-validation-keywords-for-obj)
  public struct Object: Hashable, Sendable, Codable {
    /// A dictionary of property names and their corresponding schemas.
    ///
    /// [properties](https://json-schema.org/draft/2020-12/json-schema-validation#name-a-vocabulary-for-structural-validation)
    public var properties: [Swift.String: NeedleGenerationSchema]?

    /// An array of property names that are required for the object.
    ///
    /// [required](https://json-schema.org/draft/2020-12/json-schema-validation#name-required)
    public var required: [Swift.String]?

    /// The minimum number of properties the object must have.
    ///
    /// [minProperties](https://json-schema.org/draft/2020-12/json-schema-validation#name-minproperties)
    public var minProperties: Int?

    /// The maximum number of properties the object can have.
    ///
    /// [maxProperties](https://json-schema.org/draft/2020-12/json-schema-validation#name-maxproperties)
    public var maxProperties: Int?

    /// A schema that defines constraints for additional properties not defined on the object.
    ///
    /// [additionalProperties](https://json-schema.org/draft/2020-12/json-schema-validation#name-a-vocabulary-for-structural-validation)
    public var additionalProperties: NeedleGenerationSchema?

    /// A dictionary of regex patterns and their corresponding schemas for matching property names.
    ///
    /// [patternProperties](https://json-schema.org/draft/2020-12/json-schema-validation#name-a-vocabulary-for-structural-validation)
    public var patternProperties: [Swift.String: NeedleGenerationSchema]?

    /// A schema that defines constraints for property names.
    ///
    /// [propertyNames](https://json-schema.org/draft/2020-12/json-schema-validation#name-a-vocabulary-for-structural-validation)
    public var propertyNames: NeedleGenerationSchema?

    /// A dictionary mapping property names to additional property names that are required when the
    /// key property is present in the object.
    ///
    /// [dependentRequired](https://json-schema.org/draft/2020-12/json-schema-validation#name-dependentrequired)
    public var dependentRequired: [Swift.String: [Swift.String]]?

    /// Creates an object-specific schema.
    ///
    /// - Parameters:
    ///   - properties: A dictionary of property names and their corresponding schemas.
    ///   - required: An array of required property names.
    ///   - minProperties: The minimum number of properties required.
    ///   - maxProperties: The maximum number of properties allowed.
    ///   - additionalProperties: A schema that defines constraints for additional properties not defined on the object.
    ///   - patternProperties: A dictionary of regex patterns and their corresponding schemas for matching property names.
    ///   - propertyNames: A schema that defines constraints for property names.
    ///   - dependentRequired: A dictionary mapping property names to additional property names that are required when the key property is present.
    public static func object(
      properties: [Swift.String: NeedleGenerationSchema]? = nil,
      required: [Swift.String]? = nil,
      minProperties: Int? = nil,
      maxProperties: Int? = nil,
      additionalProperties: NeedleGenerationSchema? = nil,
      patternProperties: [Swift.String: NeedleGenerationSchema]? = nil,
      propertyNames: NeedleGenerationSchema? = nil,
      dependentRequired: [Swift.String: [Swift.String]]? = nil
    ) -> Self {
      Self(
        properties: properties,
        required: required,
        minProperties: minProperties,
        maxProperties: maxProperties,
        additionalProperties: additionalProperties,
        patternProperties: patternProperties,
        propertyNames: propertyNames,
        dependentRequired: dependentRequired
      )
    }
  }

  /// Creates an object-specific schema.
  ///
  /// - Parameters:
  ///   - properties: A dictionary of property names and their corresponding schemas.
  ///   - required: An array of required property names.
  ///   - minProperties: The minimum number of properties required.
  ///   - maxProperties: The maximum number of properties allowed.
  ///   - additionalProperties: A schema that defines constraints for additional properties not defined on the object.
  ///   - patternProperties: A dictionary of regex patterns and their corresponding schemas for matching property names.
  ///   - propertyNames: A schema that defines constraints for property names.
  ///   - dependentRequired: A dictionary mapping property names to additional property names that are required when the key property is present.
  public static func object(
    properties: [Swift.String: NeedleGenerationSchema]? = nil,
    required: [Swift.String]? = nil,
    minProperties: Int? = nil,
    maxProperties: Int? = nil,
    additionalProperties: NeedleGenerationSchema? = nil,
    patternProperties: [Swift.String: NeedleGenerationSchema]? = nil,
    propertyNames: NeedleGenerationSchema? = nil,
    dependentRequired: [Swift.String: [Swift.String]]? = nil
  ) -> Self {
    .union(
      object: .object(
        properties: properties,
        required: required,
        minProperties: minProperties,
        maxProperties: maxProperties,
        additionalProperties: additionalProperties,
        patternProperties: patternProperties,
        propertyNames: propertyNames,
        dependentRequired: dependentRequired
      )
    )
  }
}
