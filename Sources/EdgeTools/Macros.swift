import OrderedCollections

// MARK: - Macros

/// Generates ``EdgeToolsGenerable`` support for a struct.
@attached(extension, conformances: EdgeToolsGenerable)
@attached(member, names: named(edgeToolsGenerationSchema), named(init))
@attached(memberAttribute)
public macro EdgeToolsGenerable(
  _ schema: EdgeToolsGenerationSchema...
) = #externalMacro(module: "EdgeToolsMacros", type: "EdgeToolsGenerableMacro")

/// Marks a stored property as ignored for ``EdgeToolsGenerationSchema`` schema synthesis.
@attached(peer)
public macro EdgeToolsIgnored() =
  #externalMacro(module: "EdgeToolsMacros", type: "EdgeToolsIgnoredMacro")

/// Overrides schema synthesis for a stored property.
@attached(peer)
public macro EdgeToolsGuide(
  key: Swift.String? = nil,
  _ schema: EdgeToolsGenerationSchema...
) = #externalMacro(module: "EdgeToolsMacros", type: "EdgeToolsGuideMacro")

@inlinable
@inline(__always)
public func _edgeToolsRequireObjectValue(
  _ value: EdgeToolsValue
) throws -> OrderedDictionary<String, EdgeToolsValue> {
  switch value {
  case .object(let object): object
  default: throw EdgeToolsValueTypeError(expected: .object, received: value.type)
  }
}

@inlinable
@inline(__always)
public func _edgeToolsValue(
  _ object: OrderedDictionary<String, EdgeToolsValue>,
  forKey key: String
) -> EdgeToolsValue {
  object[key] ?? .null
}
