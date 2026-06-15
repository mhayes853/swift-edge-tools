// MARK: - Macros

/// Generates ``NeedleGenerable`` support for a struct.
@attached(extension, conformances: NeedleGenerable)
@attached(member, names: named(needleGenerationSchema), named(init))
@attached(memberAttribute)
public macro NeedleGenerable(
  title: String? = nil,
  description: String? = nil
) = #externalMacro(module: "NeedleMacros", type: "NeedleGenerableMacro")

/// Marks a stored property as ignored for ``NeedleGenerationSchema`` schema synthesis.
@attached(peer)
public macro NeedleIgnored() =
  #externalMacro(module: "NeedleMacros", type: "NeedleIgnoredMacro")

/// Overrides schema synthesis for a stored property.
@attached(peer)
public macro NeedleGuide(
  _ schema: _NeedleGuideSchema = .inferred,
  key: Swift.String? = nil,
  description: String? = nil
) = #externalMacro(module: "NeedleMacros", type: "NeedleGuideMacro")

public struct _NeedleGuideSchema {
  public static var inferred: Self { Self() }

  public static func string(
    minLength: Int? = nil,
    maxLength: Int? = nil,
    pattern: String? = nil
  ) -> Self { Self() }

  @available(iOS 16, macOS 13, tvOS 16, watchOS 9, *)
  public static func string<Output>(
    minLength: Int? = nil,
    maxLength: Int? = nil,
    pattern: Regex<Output>? = nil
  ) -> Self { Self() }

  public static func number(
    multipleOf: Double? = nil,
    minimum: Double? = nil,
    exclusiveMinimum: Double? = nil,
    maximum: Double? = nil,
    exclusiveMaximum: Double? = nil
  ) -> Self { Self() }

  public static func integer(
    multipleOf: Int? = nil,
    minimum: Int? = nil,
    exclusiveMinimum: Int? = nil,
    maximum: Int? = nil,
    exclusiveMaximum: Int? = nil
  ) -> Self { Self() }

  public static var boolean: Self { Self() }

  public static func object(
    properties: [Swift.String: NeedleGenerationSchema]? = nil,
    required: [Swift.String]? = nil,
    minProperties: Int? = nil,
    maxProperties: Int? = nil,
    additionalProperties: NeedleGenerationSchema? = nil,
    patternProperties: [Swift.String: NeedleGenerationSchema]? = nil,
    propertyNames: NeedleGenerationSchema? = nil,
    dependentRequired: [Swift.String: [Swift.String]]? = nil
  ) -> Self { Self() }

  public static func array(
    items: NeedleGenerationSchema? = nil,
    prefixItems: [NeedleGenerationSchema]? = nil,
    minItems: Int? = nil,
    maxItems: Int? = nil,
    uniqueItems: Bool? = nil,
    contains: NeedleGenerationSchema? = nil,
    minContains: Int? = nil,
    maxContains: Int? = nil
  ) -> Self { Self() }

  public static func custom(_ schema: NeedleGenerationSchema) -> Self { Self() }
}

@inlinable
@inline(__always)
public func _needleMergeGenerationSchema(
  _ schema: NeedleGenerationSchema,
  title: String? = nil,
  description: String? = nil
) -> NeedleGenerationSchema {
  guard case .object(var object) = schema else { return schema }
  if let title {
    object.title = title
  }
  if let description {
    object.description = description
  }
  return .object(object)
}

@inlinable
@inline(__always)
public func _needleRequireObjectValue(_ value: NeedleValue) throws -> [String: NeedleValue] {
  switch value {
  case .object(let object): object
  default: throw NeedleValueTypeError(expected: .object, received: value.type)
  }
}

@inlinable
@inline(__always)
public func _needleValue(_ object: [String: NeedleValue], forKey key: String) -> NeedleValue {
  object[key] ?? .null
}
