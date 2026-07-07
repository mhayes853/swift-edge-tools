import OrderedCollections

// MARK: - Macros

/// Generates ``NeedleGenerable`` support for a struct.
@attached(extension, conformances: NeedleGenerable)
@attached(member, names: named(needleGenerationSchema), named(init))
@attached(memberAttribute)
public macro NeedleGenerable(
  _ schema: NeedleGenerationSchema...
) = #externalMacro(module: "NeedleMacros", type: "NeedleGenerableMacro")

/// Marks a stored property as ignored for ``NeedleGenerationSchema`` schema synthesis.
@attached(peer)
public macro NeedleIgnored() =
  #externalMacro(module: "NeedleMacros", type: "NeedleIgnoredMacro")

/// Overrides schema synthesis for a stored property.
@attached(peer)
public macro NeedleGuide(
  key: Swift.String? = nil,
  _ schema: NeedleGenerationSchema...
) = #externalMacro(module: "NeedleMacros", type: "NeedleGuideMacro")

@inlinable
@inline(__always)
public func _needleRequireObjectValue(
  _ value: NeedleValue
) throws -> OrderedDictionary<String, NeedleValue> {
  switch value {
  case .object(let object): object
  default: throw NeedleValueTypeError(expected: .object, received: value.type)
  }
}

@inlinable
@inline(__always)
public func _needleValue(
  _ object: OrderedDictionary<String, NeedleValue>,
  forKey key: String
) -> NeedleValue {
  object[key] ?? .null
}
