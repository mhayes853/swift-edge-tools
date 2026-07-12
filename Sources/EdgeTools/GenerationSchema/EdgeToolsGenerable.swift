#if canImport(Foundation)
  import Foundation
#endif

// MARK: - EdgeToolsGenerable

/// A type that can generate a generation schema description of itself.
public protocol EdgeToolsGenerable: ConvertibleFromEdgeToolsValue {
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

// MARK: - Scalar Types

extension String: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .string }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    guard case .string(let string) = edgeToolsValue else {
      throw EdgeToolsValueTypeError(expected: .string, received: edgeToolsValue.type)
    }
    self = string
  }
}

extension Bool: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .boolean }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    guard case .boolean(let boolean) = edgeToolsValue else {
      throw EdgeToolsValueTypeError(expected: .boolean, received: edgeToolsValue.type)
    }
    self = boolean
  }
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
}

extension Int8: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .integer }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    self = try Self.integer(from: edgeToolsValue)
  }
}

extension Int16: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .integer }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    self = try Self.integer(from: edgeToolsValue)
  }
}

extension Int32: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .integer }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    self = try Self.integer(from: edgeToolsValue)
  }
}

extension Int64: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .integer }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    self = try Self.integer(from: edgeToolsValue)
  }
}

extension Int: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .integer }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    self = try Self.integer(from: edgeToolsValue)
  }
}

extension UInt8: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    EdgeToolsGenerationSchema(.integer, .minimum(0))
  }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    self = try Self.integer(from: edgeToolsValue)
  }
}

extension UInt16: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    EdgeToolsGenerationSchema(.integer, .minimum(0))
  }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    self = try Self.integer(from: edgeToolsValue)
  }
}

extension UInt32: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    EdgeToolsGenerationSchema(.integer, .minimum(0))
  }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    self = try Self.integer(from: edgeToolsValue)
  }
}

extension UInt64: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    EdgeToolsGenerationSchema(.integer, .minimum(0))
  }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    self = try Self.integer(from: edgeToolsValue)
  }
}

extension UInt: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    EdgeToolsGenerationSchema(.integer, .minimum(0))
  }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    self = try Self.integer(from: edgeToolsValue)
  }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
extension Int128: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .integer }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    self = try Self.integer(from: edgeToolsValue)
  }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
extension UInt128: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    EdgeToolsGenerationSchema(.integer, .minimum(0))
  }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    self = try Self.integer(from: edgeToolsValue)
  }
}

// MARK: - Foundation

#if canImport(Foundation)
  extension Data: EdgeToolsGenerable {
    public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .string }

    public init(edgeToolsValue: EdgeToolsValue) throws {
      guard case .string(let string) = edgeToolsValue else {
        throw EdgeToolsValueTypeError(expected: .string, received: edgeToolsValue.type)
      }
      self = Data(string.utf8)
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
      default:
        throw EdgeToolsValueTypeError(expected: .number, received: edgeToolsValue.type)
      }
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

extension Dictionary: EdgeToolsGenerable
where Key == String, Value: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    EdgeToolsGenerationSchema(.type(.object), .additionalProperties(Value.edgeToolsGenerationSchema))
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

#if canImport(CoreGraphics)
  import CoreGraphics

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
