#if canImport(Foundation)
  import Foundation
#endif

// MARK: - NeedleGenerable

/// A type that can generate a Needle generation schema description of itself.
public protocol NeedleGenerable: ConvertibleFromNeedleValue {
  /// The Needle generation schema describing this type.
  static var needleGenerationSchema: NeedleGenerationSchema { get }
}

// MARK: - ConvertibleFromNeedleValue

public protocol ConvertibleFromNeedleValue {
  associatedtype NeedleConversionFailure: Error

  /// Creates this type from a ``NeedleValue``.
  init(needleValue: NeedleValue) throws(NeedleConversionFailure)
}

extension NeedleValue: ConvertibleFromNeedleValue {
  public init(needleValue: NeedleValue) {
    self = needleValue
  }
}

// MARK: - Scalar Types

extension String: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .string }

  public init(needleValue: NeedleValue) throws {
    guard case .string(let string) = needleValue else {
      throw NeedleValueTypeError(expected: .string, received: needleValue.type)
    }
    self = string
  }
}

extension Bool: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .boolean }

  public init(needleValue: NeedleValue) throws {
    guard case .boolean(let boolean) = needleValue else {
      throw NeedleValueTypeError(expected: .boolean, received: needleValue.type)
    }
    self = boolean
  }
}

extension Double: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .number }

  public init(needleValue: NeedleValue) throws {
    switch needleValue {
    case .number(let number):
      self = number
    case .integer(let integer):
      self = Double(integer)
    default:
      throw NeedleValueTypeError(expected: .number, received: needleValue.type)
    }
  }
}

extension Float: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .number }

  public init(needleValue: NeedleValue) throws {
    switch needleValue {
    case .number(let number):
      self = Float(number)
    case .integer(let integer):
      self = Float(integer)
    default:
      throw NeedleValueTypeError(expected: .number, received: needleValue.type)
    }
  }
}

extension Int8: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer }

  public init(needleValue: NeedleValue) throws {
    self = try Self.integer(from: needleValue)
  }
}

extension Int16: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer }

  public init(needleValue: NeedleValue) throws {
    self = try Self.integer(from: needleValue)
  }
}

extension Int32: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer }

  public init(needleValue: NeedleValue) throws {
    self = try Self.integer(from: needleValue)
  }
}

extension Int64: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer }

  public init(needleValue: NeedleValue) throws {
    self = try Self.integer(from: needleValue)
  }
}

extension Int: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer }

  public init(needleValue: NeedleValue) throws {
    self = try Self.integer(from: needleValue)
  }
}

extension UInt8: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema {
    NeedleGenerationSchema(.integer, .minimum(0))
  }

  public init(needleValue: NeedleValue) throws {
    self = try Self.integer(from: needleValue)
  }
}

extension UInt16: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema {
    NeedleGenerationSchema(.integer, .minimum(0))
  }

  public init(needleValue: NeedleValue) throws {
    self = try Self.integer(from: needleValue)
  }
}

extension UInt32: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema {
    NeedleGenerationSchema(.integer, .minimum(0))
  }

  public init(needleValue: NeedleValue) throws {
    self = try Self.integer(from: needleValue)
  }
}

extension UInt64: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema {
    NeedleGenerationSchema(.integer, .minimum(0))
  }

  public init(needleValue: NeedleValue) throws {
    self = try Self.integer(from: needleValue)
  }
}

extension UInt: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema {
    NeedleGenerationSchema(.integer, .minimum(0))
  }

  public init(needleValue: NeedleValue) throws {
    self = try Self.integer(from: needleValue)
  }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
extension Int128: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer }

  public init(needleValue: NeedleValue) throws {
    self = try Self.integer(from: needleValue)
  }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
extension UInt128: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema {
    NeedleGenerationSchema(.integer, .minimum(0))
  }

  public init(needleValue: NeedleValue) throws {
    self = try Self.integer(from: needleValue)
  }
}

// MARK: - Foundation

#if canImport(Foundation)
  extension Data: NeedleGenerable {
    public static var needleGenerationSchema: NeedleGenerationSchema { .string }

    public init(needleValue: NeedleValue) throws {
      guard case .string(let string) = needleValue else {
        throw NeedleValueTypeError(expected: .string, received: needleValue.type)
      }
      self = Data(string.utf8)
    }
  }

  extension Decimal: NeedleGenerable {
    public static var needleGenerationSchema: NeedleGenerationSchema { .number }

    public init(needleValue: NeedleValue) throws {
      switch needleValue {
      case .number(let number):
        self = Decimal(number)
      case .integer(let integer):
        self = Decimal(integer)
      default:
        throw NeedleValueTypeError(expected: .number, received: needleValue.type)
      }
    }
  }
#endif

// MARK: - Generic Containers

extension Array: NeedleGenerable where Element: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema {
    NeedleGenerationSchema(.type(.array), .items(Element.needleGenerationSchema))
  }
}

extension Array: ConvertibleFromNeedleValue where Element: ConvertibleFromNeedleValue {
  public init(needleValue: NeedleValue) throws {
    guard case .array(let array) = needleValue else {
      throw NeedleValueTypeError(expected: .array, received: needleValue.type)
    }
    self = try array.map { try Element(needleValue: $0) }
  }
}

extension Dictionary: NeedleGenerable
where Key == String, Value: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema {
    NeedleGenerationSchema(.type(.object), .additionalProperties(Value.needleGenerationSchema))
  }
}

extension Dictionary: ConvertibleFromNeedleValue
where Key == String, Value: ConvertibleFromNeedleValue {
  public init(needleValue: NeedleValue) throws {
    guard case .object(let object) = needleValue else {
      throw NeedleValueTypeError(expected: .object, received: needleValue.type)
    }
    self = try Dictionary(
      uniqueKeysWithValues: object.map { key, value in
        (key, try Value(needleValue: value))
      }
    )
  }
}

extension Optional: NeedleGenerable where Wrapped: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema {
    Wrapped.needleGenerationSchema.nullable()
  }
}

extension Optional: ConvertibleFromNeedleValue where Wrapped: ConvertibleFromNeedleValue {
  public init(needleValue: NeedleValue) throws {
    switch needleValue {
    case .null:
      self = nil
    default:
      self = try Wrapped(needleValue: needleValue)
    }
  }
}

#if canImport(CoreGraphics)
  import CoreGraphics

  extension CGFloat: NeedleGenerable {
    public static var needleGenerationSchema: NeedleGenerationSchema { .number }

    public init(needleValue: NeedleValue) throws {
      switch needleValue {
      case .number(let number):
        self = CGFloat(number)
      case .integer(let integer):
        self = CGFloat(integer)
      default:
        throw NeedleValueTypeError(expected: .number, received: needleValue.type)
      }
    }
  }
#endif

// MARK: - NeedleValueTypeError

public struct NeedleValueTypeError: Error, Hashable, Sendable {
  public let expected: NeedleGenerationSchema.ValueType
  public let received: NeedleGenerationSchema.ValueType

  public init(
    expected: NeedleGenerationSchema.ValueType,
    received: NeedleGenerationSchema.ValueType
  ) {
    self.expected = expected
    self.received = received
  }
}

// MARK: - Helpers

extension FixedWidthInteger {
  fileprivate static func integer(from needleValue: NeedleValue) throws -> Self {
    guard case .integer(let integer) = needleValue else {
      throw NeedleValueTypeError(expected: .integer, received: needleValue.type)
    }
    guard let value = Self(exactly: integer) else {
      throw NeedleValueTypeError(expected: .integer, received: needleValue.type)
    }
    return value
  }
}
