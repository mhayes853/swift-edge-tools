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
/// Also generates a stream-parseable `Partial` for nongeneric structs and enums with one
/// `@StreamParseableDefault` case. A nested user-declared `Partial` leaves stream parsing
/// to a manually provided conformance.
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
