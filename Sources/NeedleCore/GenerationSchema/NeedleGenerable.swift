#if canImport(Foundation)
  import Foundation
#endif

// MARK: - NeedleGenerable

/// A type that can generate a Needle generation schema description of itself.
public protocol NeedleGenerable {
  /// The Needle generation schema describing this type.
  static var needleGenerationSchema: NeedleGenerationSchema { get }
}

// MARK: - Scalar Types

extension String: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .string() }
}

extension Bool: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .bool() }
}

extension Double: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .number() }
}

extension Float: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .number() }
}

extension Int8: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer() }
}

extension Int16: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer() }
}

extension Int32: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer() }
}

extension Int64: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer() }
}

extension Int: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer() }
}

extension UInt8: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer(minimum: 0) }
}

extension UInt16: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer(minimum: 0) }
}

extension UInt32: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer(minimum: 0) }
}

extension UInt64: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer(minimum: 0) }
}

extension UInt: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer(minimum: 0) }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
extension Int128: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer() }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
extension UInt128: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema { .integer(minimum: 0) }
}

// MARK: - Foundation

#if canImport(Foundation)
  extension Data: NeedleGenerable {
    public static var needleGenerationSchema: NeedleGenerationSchema { .string() }
  }

  extension Decimal: NeedleGenerable {
    public static var needleGenerationSchema: NeedleGenerationSchema { .number() }
  }
#endif

// MARK: - Generic Containers

extension Array: NeedleGenerable where Element: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema {
    .array(items: .schemaForAll(Element.needleGenerationSchema))
  }
}

extension Dictionary: NeedleGenerable
where Key == String, Value: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema {
    .object(additionalProperties: Value.needleGenerationSchema)
  }
}

extension Optional: NeedleGenerable where Wrapped: NeedleGenerable {
  public static var needleGenerationSchema: NeedleGenerationSchema {
    .object(anyOf: [Wrapped.needleGenerationSchema, .null()])
  }
}

#if canImport(CoreGraphics)
  import CoreGraphics

  extension CGFloat: NeedleGenerable {
    public static var needleGenerationSchema: NeedleGenerationSchema { .number() }
  }
#endif

public struct NeedleValueTypeError: Error {
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
