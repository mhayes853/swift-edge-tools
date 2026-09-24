import EdgeToolsCore
import OrderedCollections
import StreamParsing

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
