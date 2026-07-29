import OrderedCollections

#if Foundation
  import _EdgeToolsFoundation
#endif

#if canImport(CoreGraphics)
  import CoreGraphics
#endif

// MARK: - EdgeToolsGenerable

/// A type that can generate a generation schema description of itself.
public protocol EdgeToolsGenerable: ConvertibleFromEdgeToolsValue, ConvertibleToEdgeToolsValue {
  /// The generation schema describing this type.
  static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { get }
}

// MARK: - ConvertibleFromEdgeToolsValue

public protocol ConvertibleFromEdgeToolsValue {
  associatedtype EdgeToolsConversionFailure: Error

  /// Creates this type from a ``EdgeToolsValue``.
  init(edgeToolsValue: EdgeToolsValue) throws(EdgeToolsConversionFailure)
}

extension EdgeToolsValue: ConvertibleFromEdgeToolsValue {
  public init(edgeToolsValue: EdgeToolsValue) {
    self = edgeToolsValue
  }
}

// MARK: - ConvertibleToEdgeToolsValue

/// A type that can produce an ``EdgeToolsValue`` representation of itself.
public protocol ConvertibleToEdgeToolsValue {
  /// An ``EdgeToolsValue`` representation of this value.
  var edgeToolsValue: EdgeToolsValue { get }
}

extension EdgeToolsValue: ConvertibleToEdgeToolsValue {
  public var edgeToolsValue: EdgeToolsValue { self }
}

// MARK: - EdgeToolsValue

extension EdgeToolsValue: EdgeToolsGenerable {
  /// The universal generation schema: a JSON Schema `true`, which accepts any
  /// well-formed ``EdgeToolsValue``.
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .universal }
}

// MARK: - Scalar Types

extension String: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .string }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    guard case .string(let string) = edgeToolsValue else {
      throw EdgeToolsValueTypeError(expected: .string, received: edgeToolsValue.type)
    }
    self = string
  }

  public var edgeToolsValue: EdgeToolsValue { .string(self) }
}

extension Bool: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .boolean }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    guard case .boolean(let boolean) = edgeToolsValue else {
      throw EdgeToolsValueTypeError(expected: .boolean, received: edgeToolsValue.type)
    }
    self = boolean
  }

  public var edgeToolsValue: EdgeToolsValue { .boolean(self) }
}

extension Double: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .number }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    switch edgeToolsValue {
    case .number(let number):
      self = number
    case .integer(let integer):
      self = Double(integer)
    default:
      throw EdgeToolsValueTypeError(expected: .number, received: edgeToolsValue.type)
    }
  }

  public var edgeToolsValue: EdgeToolsValue { .number(self) }
}

extension Float: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .number }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    switch edgeToolsValue {
    case .number(let number):
      self = Float(number)
    case .integer(let integer):
      self = Float(integer)
    default:
      throw EdgeToolsValueTypeError(expected: .number, received: edgeToolsValue.type)
    }
  }

  public var edgeToolsValue: EdgeToolsValue { .number(Double(self)) }
}

extension EdgeToolsGenerable where Self: FixedWidthInteger, Self: SignedInteger {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .integer }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    self = try Self.integer(from: edgeToolsValue)
  }

  public var edgeToolsValue: EdgeToolsValue { .integer(Int(self)) }
}

extension EdgeToolsGenerable where Self: FixedWidthInteger, Self: UnsignedInteger {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    EdgeToolsGenerationSchema(.integer, .minimum(0))
  }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    self = try Self.integer(from: edgeToolsValue)
  }

  public var edgeToolsValue: EdgeToolsValue {
    let asInt = Int(truncatingIfNeeded: self)
    precondition(
      asInt >= 0,
      "\(Self.self) value \(self) exceeds Int.max and cannot be encoded as an EdgeToolsValue integer."
    )
    return .integer(asInt)
  }
}

extension Int8: EdgeToolsGenerable {}
extension Int16: EdgeToolsGenerable {}
extension Int32: EdgeToolsGenerable {}
extension Int64: EdgeToolsGenerable {}
extension Int: EdgeToolsGenerable {}
extension UInt8: EdgeToolsGenerable {}
extension UInt16: EdgeToolsGenerable {}
extension UInt32: EdgeToolsGenerable {}
extension UInt64: EdgeToolsGenerable {}
extension UInt: EdgeToolsGenerable {}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
extension Int128: EdgeToolsGenerable {}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
extension UInt128: EdgeToolsGenerable {}

// MARK: - Foundation

#if Foundation
  extension Data: EdgeToolsGenerable {
    public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .string }

    public init(edgeToolsValue: EdgeToolsValue) throws {
      guard case .string(let string) = edgeToolsValue else {
        throw EdgeToolsValueTypeError(expected: .string, received: edgeToolsValue.type)
      }
      self = Data(string.utf8)
    }

    public var edgeToolsValue: EdgeToolsValue {
      .string(String(decoding: self, as: UTF8.self))
    }
  }

  extension Decimal: EdgeToolsGenerable {
    public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .number }

    public init(edgeToolsValue: EdgeToolsValue) throws {
      switch edgeToolsValue {
      case .number(let number):
        self = Decimal(number)
      case .integer(let integer):
        self = Decimal(integer)
      case .string(let string):
        guard let decimal = Decimal(string: string, locale: Locale(identifier: "en_US_POSIX")) else {
          throw EdgeToolsValueTypeError(expected: .number, received: .string)
        }
        self = decimal
      default:
        throw EdgeToolsValueTypeError(expected: .number, received: edgeToolsValue.type)
      }
    }

    public var edgeToolsValue: EdgeToolsValue {
      .string("\(self)")
    }
  }
#endif

// MARK: - Generic Containers

extension Array: EdgeToolsGenerable where Element: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    EdgeToolsGenerationSchema(.type(.array), .items(Element.edgeToolsGenerationSchema))
  }
}

extension Array: ConvertibleFromEdgeToolsValue where Element: ConvertibleFromEdgeToolsValue {
  public init(edgeToolsValue: EdgeToolsValue) throws {
    guard case .array(let array) = edgeToolsValue else {
      throw EdgeToolsValueTypeError(expected: .array, received: edgeToolsValue.type)
    }
    self = try array.map { try Element(edgeToolsValue: $0) }
  }
}

extension Array: ConvertibleToEdgeToolsValue where Element: ConvertibleToEdgeToolsValue {
  public var edgeToolsValue: EdgeToolsValue { .array(self.map(\.edgeToolsValue)) }
}

extension Dictionary: EdgeToolsGenerable
where Key == String, Value: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    EdgeToolsGenerationSchema(
      .type(.object),
      .additionalProperties(Value.edgeToolsGenerationSchema)
    )
  }
}

extension Dictionary: ConvertibleFromEdgeToolsValue
where Key == String, Value: ConvertibleFromEdgeToolsValue {
  public init(edgeToolsValue: EdgeToolsValue) throws {
    guard case .object(let object) = edgeToolsValue else {
      throw EdgeToolsValueTypeError(expected: .object, received: edgeToolsValue.type)
    }
    self = try Dictionary(
      uniqueKeysWithValues: object.map { key, value in
        (key, try Value(edgeToolsValue: value))
      }
    )
  }
}

extension Dictionary: ConvertibleToEdgeToolsValue
where Key == String, Value: ConvertibleToEdgeToolsValue {
  public var edgeToolsValue: EdgeToolsValue {
    .object(
      OrderedDictionary(
        uniqueKeysWithValues: self.map { key, value in (key, value.edgeToolsValue) }
      )
    )
  }
}

extension Optional: EdgeToolsGenerable where Wrapped: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    Wrapped.edgeToolsGenerationSchema.nullable()
  }
}

extension Optional: ConvertibleFromEdgeToolsValue where Wrapped: ConvertibleFromEdgeToolsValue {
  public init(edgeToolsValue: EdgeToolsValue) throws {
    switch edgeToolsValue {
    case .null:
      self = nil
    default:
      self = try Wrapped(edgeToolsValue: edgeToolsValue)
    }
  }
}

extension Optional: ConvertibleToEdgeToolsValue where Wrapped: ConvertibleToEdgeToolsValue {
  public var edgeToolsValue: EdgeToolsValue {
    switch self {
    case .none: .null
    case .some(let wrapped): wrapped.edgeToolsValue
    }
  }
}

#if canImport(CoreGraphics)
  extension CGFloat: EdgeToolsGenerable {
    public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .number }

    public init(edgeToolsValue: EdgeToolsValue) throws {
      switch edgeToolsValue {
      case .number(let number):
        self = CGFloat(number)
      case .integer(let integer):
        self = CGFloat(integer)
      default:
        throw EdgeToolsValueTypeError(expected: .number, received: edgeToolsValue.type)
      }
    }

    public var edgeToolsValue: EdgeToolsValue { .number(Double(self)) }
  }
#endif

// MARK: - EdgeToolsValueTypeError

public struct EdgeToolsValueTypeError: Error, Hashable, Sendable {
  public let expected: EdgeToolsGenerationSchema.ValueType
  public let received: EdgeToolsGenerationSchema.ValueType

  public init(
    expected: EdgeToolsGenerationSchema.ValueType,
    received: EdgeToolsGenerationSchema.ValueType
  ) {
    self.expected = expected
    self.received = received
  }
}

// MARK: - Helpers

extension FixedWidthInteger {
  fileprivate static func integer(from edgeToolsValue: EdgeToolsValue) throws -> Self {
    guard case .integer(let integer) = edgeToolsValue else {
      throw EdgeToolsValueTypeError(expected: .integer, received: edgeToolsValue.type)
    }
    guard let value = Self(exactly: integer) else {
      throw EdgeToolsValueTypeError(expected: .integer, received: edgeToolsValue.type)
    }
    return value
  }
}

