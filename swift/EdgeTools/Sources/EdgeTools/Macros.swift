import EdgeToolsCore
import OrderedCollections
import StreamParsing

// MARK: - Macros

/// Generates ``EdgeToolsGenerable`` support for a struct or associated-value enum.
///
/// Enum values use an externally tagged object representation. Every case payload is an object;
/// labeled associated values use their labels as keys and unlabeled values use positional keys
/// such as `_0` and `_1`.
///
/// Also generates a stream-parseable `Partial` for structs and nongeneric enums with one
/// `@StreamParseableDefault` case. Structs may be generic or nested in generic types. Generic
/// members parsed directly must conform to `StreamParseable`, with `EdgeToolsGenerable` partials.
/// A nested user-declared `Partial` leaves stream parsing to a manually provided conformance.
///
/// Generic partials do not automatically conform to `Sendable`. Declare that conformance when
/// their member partials are sendable:
///
/// ```swift
/// @EdgeToolsGenerable
/// struct Page<Item: EdgeToolsGenerable & StreamParseable & Sendable>: Sendable
/// where Item.Partial: EdgeToolsGenerable {
///   var items: [Item]
/// }
/// extension Page.Partial: Sendable where Item.Partial: Sendable {}
/// ```
///
/// `partialStrings` selects `StreamString` or Swift `String` storage for string leaves in the
/// generated partial. Nested generable types select their own storage. `keyDecodingStrategy`
/// derives property, enum case, and associated value keys for both generation and parsing;
/// explicit `@EdgeToolsGuide(key:)` keys are preserved. Custom strategies must return stable keys.
/// `schemaCache` selects the cache for streaming schemas. Cache and custom strategy expressions
/// are evaluated inside the generated `Partial`, so qualify references rather than using `Self`.
@attached(extension, conformances: EdgeToolsGenerable, StreamParseable, names: arbitrary)
@attached(
  member,
  names: named(edgeToolsGenerationSchema),
  named(init),
  named(edgeToolsValue)
)
public macro EdgeToolsGenerable(
  _ schema: EdgeToolsGenerationSchema...,
  partialStrings: PartialStringStorage = .streamString,
  keyDecodingStrategy: StreamKeyDecodingStrategy = .useDefaultKeys,
  schemaCache: StreamSchemaCache = .shared
) =
  #externalMacro(module: "EdgeToolsMacros", type: "EdgeToolsGenerableMacro")

/// Marks a stored property as ignored for ``EdgeToolsGenerationSchema`` schema synthesis.
@attached(peer)
public macro EdgeToolsIgnored() =
  #externalMacro(module: "EdgeToolsMacros", type: "EdgeToolsIgnoredMacro")

/// Overrides schema synthesis and streaming storage for a stored property.
///
/// `partialStrings` overrides the enclosing type's string storage choice. `initialCapacity`
/// reserves storage for strings, arrays, or dictionaries when parsing begins; it is a
/// nonnegative integer literal hint, measured in decoded UTF-8 bytes or container elements.
@attached(peer)
public macro EdgeToolsGuide(
  key: Swift.String? = nil,
  initialCapacity: Swift.Int? = nil,
  partialStrings: PartialStringStorage? = nil,
  _ schema: EdgeToolsGenerationSchema...
) = #externalMacro(module: "EdgeToolsMacros", type: "EdgeToolsGuideMacro")

/// Converts a property's JSON source representation after its value finishes parsing.
///
/// The strategy's `Value` must match the property's unwrapped type. Its `Source` supplies the
/// generation schema and value representation; schema fragments constrain that source.
/// The generated partial stores `ConvertedPartial<Conversion>`, exposing the incremental
/// `source`, cached `value`, and `conversionError`. Nonoptional properties require a default
/// for `init(orInitial:)`. Capacity and string storage overrides are unavailable because the
/// strategy defines its source storage.
@attached(peer)
public macro EdgeToolsGuide<Conversion: StreamCompletedValueConversion>(
  key: Swift.String? = nil,
  completedConversion: Conversion.Type,
  _ schema: EdgeToolsGenerationSchema...
) = #externalMacro(module: "EdgeToolsMacros", type: "EdgeToolsGuideMacro")
where Conversion.Source: EdgeToolsGenerable

@inlinable
@inline(always)
public func _edgeToolsRequireObjectValue(
  _ value: EdgeToolsValue
) throws -> OrderedDictionary<String, EdgeToolsValue> {
  switch value {
  case .object(let object): object
  default: throw EdgeToolsValueTypeError(expected: .object, received: value.type)
  }
}

@inlinable
@inline(always)
public func _edgeToolsRequireObjectValue(
  _ value: EdgeToolsValue,
  keys: [String]
) throws -> OrderedDictionary<String, EdgeToolsValue> {
  let object = try _edgeToolsRequireObjectValue(value)
  guard keys.allSatisfy({ object[$0] != nil }) else {
    throw EdgeToolsObjectKeysError(
      expected: keys,
      received: Array(object.keys)
    )
  }
  return object
}

@inlinable
@inline(always)
public func _edgeToolsValue(
  _ object: OrderedDictionary<String, EdgeToolsValue>,
  forKey key: String
) -> EdgeToolsValue {
  object[key] ?? .null
}

/// Decodes partial storage while preserving a nullable root's explicit null and missing fields.
public func _edgeToolsPartialValue<Partial: EdgeToolsGenerable & StreamParseableRoot>(
  _ value: EdgeToolsValue?,
  as type: Partial.Type
) throws -> Partial? {
  guard let value else {
    return nil
  }
  if value == .null {
    return Partial._streamNullValue
  }
  return try Partial(edgeToolsValue: value)
}

@inlinable
@inline(always)
public func _edgeToolsBuildObjectValue(
  _ entries: (key: String, value: EdgeToolsValue?)...
) -> EdgeToolsValue {
  var object = OrderedDictionary<String, EdgeToolsValue>()
  for entry in entries {
    if let value = entry.value {
      object[entry.key] = value
    }
  }
  return .object(object)
}

// MARK: - Macro Conversion Errors

/// An error thrown when an object does not contain its required keys.
public struct EdgeToolsObjectKeysError: Error, Hashable, Sendable {
  /// The required object keys.
  public let expected: [String]
  /// The received object keys.
  public let received: [String]

  public init(expected: [String], received: [String]) {
    self.expected = expected
    self.received = received
  }
}

/// An error thrown when an enum representation contains an unknown case name.
public struct EdgeToolsUnknownEnumCaseError: Error, Hashable, Sendable {
  /// The enum type name.
  public let typeName: String
  /// The unrecognized case name.
  public let caseName: String

  public init(typeName: String, caseName: String) {
    self.typeName = typeName
    self.caseName = caseName
  }
}
