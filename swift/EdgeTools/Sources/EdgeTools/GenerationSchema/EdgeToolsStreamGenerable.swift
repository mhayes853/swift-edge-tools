import EdgeToolsCore
import OrderedCollections
import StreamParsing

// MARK: - Completed Conversions

extension StreamCompletedValueConversion where Source: EdgeToolsGenerable {
  /// Decodes and converts a completed source value, propagating decoding and conversion errors.
  public static func value(edgeToolsValue: EdgeToolsValue) throws -> Value {
    try Self.value(fromCompletedSource: Source(edgeToolsValue: edgeToolsValue))
  }
}

extension ConvertedPartial: EdgeToolsGenerable where Strategy.Source: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    Strategy.Source.edgeToolsGenerationSchema
  }
}

extension ConvertedPartial: ConvertibleFromEdgeToolsValue
where Strategy.Source: EdgeToolsGenerable {
  /// Restores a completed source, caching its conversion and preserving its representation.
  /// A value snapshot carries no completion state, so restoring an unfinished source can fail conversion.
  public init(edgeToolsValue: EdgeToolsValue) throws {
    self.init(completedSource: try Strategy.Source(edgeToolsValue: edgeToolsValue))
    if let error = self.conversionError {
      throw error
    }
  }
}

extension ConvertedPartial: ConvertibleToEdgeToolsValue
where Strategy.Source: ConvertibleToEdgeToolsValue {
  public var edgeToolsValue: EdgeToolsValue { self.source.edgeToolsValue }
}

// MARK: - Stream Scalars

extension StreamString: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .string }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    self.init(try String(edgeToolsValue: edgeToolsValue))
  }

  public var edgeToolsValue: EdgeToolsValue { .string(String(self)) }
}

extension StreamEmptyObject: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    EdgeToolsGenerationSchema(.type(.object))
  }

  public init(edgeToolsValue: EdgeToolsValue) throws {
    _ = try _edgeToolsRequireObjectValue(edgeToolsValue)
    self.init()
  }

  public var edgeToolsValue: EdgeToolsValue { .object([:]) }
}

// MARK: - Stream Collections

extension StreamArray: EdgeToolsGenerable where Element: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    EdgeToolsGenerationSchema(.type(.array), .items(Element.edgeToolsGenerationSchema))
  }
}

extension StreamArray: ConvertibleFromEdgeToolsValue where Element: ConvertibleFromEdgeToolsValue {
  public init(edgeToolsValue: EdgeToolsValue) throws {
    guard case .array(let values) = edgeToolsValue else {
      throw EdgeToolsValueTypeError(expected: .array, received: edgeToolsValue.type)
    }
    self.init(try values.map { try Element(edgeToolsValue: $0) })
  }
}

extension StreamArray: ConvertibleToEdgeToolsValue where Element: ConvertibleToEdgeToolsValue {
  public var edgeToolsValue: EdgeToolsValue {
    .array(self.map(\.edgeToolsValue))
  }
}

extension StreamDictionary: EdgeToolsGenerable where Value: EdgeToolsGenerable {
  public static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
    EdgeToolsGenerationSchema(
      .type(.object),
      .additionalProperties(Value.edgeToolsGenerationSchema)
    )
  }
}

extension StreamDictionary: ConvertibleFromEdgeToolsValue
where Value: ConvertibleFromEdgeToolsValue {
  public init(edgeToolsValue: EdgeToolsValue) throws {
    guard case .object(let values) = edgeToolsValue else {
      throw EdgeToolsValueTypeError(expected: .object, received: edgeToolsValue.type)
    }
    self.init(
      try values.map {
        (key: $0.key, value: try Value(edgeToolsValue: $0.value))
      }
    )
  }
}

extension StreamDictionary: ConvertibleToEdgeToolsValue where Value: ConvertibleToEdgeToolsValue {
  public var edgeToolsValue: EdgeToolsValue {
    .object(
      OrderedDictionary(
        uniqueKeysWithValues: self.map {
          ($0.key, $0.value.edgeToolsValue)
        }
      )
    )
  }
}
