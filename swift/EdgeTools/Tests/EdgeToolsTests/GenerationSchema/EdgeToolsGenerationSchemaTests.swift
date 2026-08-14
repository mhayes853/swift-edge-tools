import CustomDump
import EdgeTools
import Foundation
import Testing

@Suite
struct `EdgeToolsEncoding tests` {
  @Test
  func `Generable Types Expose Extraction Tool Definitions`() {
    expectNoDifference(MacroEncodingUser.extractionToolDefinition.name, "macro_encoding_user")
    expectNoDifference(MacroEncodingAddress.extractionToolDefinition.name, "macro_encoding_address")
    expectNoDifference(String.extractionToolDefinition.name, "string")
    expectNoDifference([String].extractionToolDefinition.name, "array")
    expectNoDifference(CustomExtractionName.extractionToolDefinition.name, "receipt")
    expectNoDifference(
      CustomExtractionName.extractionToolDefinition.description,
      "Extract a receipt."
    )
    expectNoDifference(
      CustomExtractionName.extractionToolDefinition.includesSchemaInInstructions,
      false
    )
  }

  @Test
  func `Data Encodes To UTF8 String Value`() {
    let data = Data("blob".utf8)
    expectNoDifference(data.edgeToolsValue, .string("blob"))
  }

  @Test
  func `Decimal Encodes To String Value`() {
    expectNoDifference(Decimal(1.5).edgeToolsValue, .string("1.5"))
    expectNoDifference(
      Decimal(string: "12345.67890123456789")!.edgeToolsValue,
      .string("12345.67890123456789")
    )
  }

  #if canImport(CoreGraphics)
    @Test
    func `CGFloat Encodes To Number Value`() {
      expectNoDifference(CGFloat(2.5).edgeToolsValue, .number(2.5))
    }
  #endif

  @Test
  func `Optional Encodes To Null Or Inner Value`() {
    expectNoDifference(String?.none.edgeToolsValue, .null)
    expectNoDifference(String?.some("blob").edgeToolsValue, .string("blob"))
    expectNoDifference(Int?.some(42).edgeToolsValue, .integer(42))
  }

  @Test
  func `Macro Generated Type Round Trips Through EdgeToolsValue`() throws {
    let value: EdgeToolsValue = [
      "name": "Blob",
      "display_age": 42,
      "nickname": "blob",
      "tags": ["swift", "tools"],
      "metadata": ["role": "admin"],
      "address": ["city": "Brooklyn"]
    ]
    let user = try MacroEncodingUser(edgeToolsValue: value)

    let encoded = user.edgeToolsValue
    let decoded = try MacroEncodingUser(edgeToolsValue: encoded)
    expectNoDifference(decoded, user)
    expectNoDifference(decoded.ignoredDefault, "default")
  }

  @Test
  func `Macro Generated Enum Round Trips Through EdgeToolsValue`() throws {
    let actions: [MacroEncodingAction] = [
      .store(try MacroEncodingAddress(edgeToolsValue: ["city": "Brooklyn"])),
      .move(12.5, 8),
      .search(query: "swift", limit: nil),
      .replace("old", with: "new")
    ]
    let expectedValues: [EdgeToolsValue] = [
      ["store": ["_0": ["city": "Brooklyn"]]],
      ["move": ["_0": 12.5, "_1": 8.0]],
      ["search": ["query": "swift", "limit": .null]],
      ["replace": ["_0": "old", "with": "new"]]
    ]

    let encoded = actions.map { $0.edgeToolsValue }
    expectNoDifference(encoded, expectedValues)
    expectNoDifference(
      try expectedValues.map(MacroEncodingAction.init(edgeToolsValue:)),
      actions
    )
  }

  @Test
  func `Macro Generated Enum Requires Known Cases And Payload Keys`() throws {
    #expect(throws: EdgeToolsUnknownEnumCaseError.self) {
      try MacroEncodingAction(edgeToolsValue: [:])
    }
    #expect(throws: EdgeToolsUnknownEnumCaseError.self) {
      try MacroEncodingAction(edgeToolsValue: ["unknown": ["_0": 1]])
    }
    #expect(throws: EdgeToolsObjectKeysError.self) {
      try MacroEncodingAction(edgeToolsValue: ["move": ["_0": 1]])
    }
    expectNoDifference(
      try MacroEncodingAction(
        edgeToolsValue: ["move": ["_0": 1, "_1": 2, "ignored": 3]]
      ),
      .move(1, 2)
    )
    expectNoDifference(
      try MacroEncodingAction(
        edgeToolsValue: ["move": ["_0": 1, "_1": 2], "ignored": .null]
      ),
      .move(1, 2)
    )
  }

  @Test
  func `Macro Generated Enum Uses AnyOf Object Cases`() {
    let expected = EdgeToolsGenerationSchema(
      .anyOf([
        enumCaseSchema("store", properties: ["_0": MacroEncodingAddress.edgeToolsGenerationSchema]),
        enumCaseSchema("move", properties: ["_0": .number, "_1": .number]),
        enumCaseSchema(
          "search",
          properties: ["query": .string, "limit": .integer.nullable()]
        ),
        enumCaseSchema("replace", properties: ["_0": .string, "with": .string])
      ])
    )

    expectNoDifference(MacroEncodingAction.edgeToolsGenerationSchema, expected)
  }

  @Test
  func `Macro Generated Encoding Decodes Escaped Key Overrides`() throws {
    let value = try EscapedGuideKey(edgeToolsValue: ["line\nbreak": "blob"])

    guard case .object(let object) = value.edgeToolsValue else {
      Issue.record("Expected object value.")
      return
    }
    expectNoDifference(Array(object.keys), ["line\nbreak"])
  }

  @Test
  func `Macro Generated Encoding Honors Key Overrides`() throws {
    let value: EdgeToolsValue = [
      "name": "Blob",
      "display_age": 42,
      "tags": [],
      "metadata": [:],
      "address": ["city": "Brooklyn"]
    ]
    let user = try MacroEncodingUser(edgeToolsValue: value)

    guard case .object(let object) = user.edgeToolsValue else {
      Issue.record("Expected object value.")
      return
    }
    expectNoDifference(
      Array(object.keys),
      ["name", "display_age", "tags", "metadata", "address"]
    )
    expectNoDifference(object["display_age"], .integer(42))
  }

  @Test
  func `Macro Generated Encoding Omits Nil Optionals And Ignored Properties`() throws {
    let value: EdgeToolsValue = [
      "name": "Blob",
      "display_age": 1,
      "tags": [],
      "metadata": [:],
      "address": ["city": "X"]
    ]
    let user = try MacroEncodingUser(edgeToolsValue: value)

    guard case .object(let object) = user.edgeToolsValue else {
      Issue.record("Expected object value.")
      return
    }
    expectNoDifference(object["nickname"], nil)
    expectNoDifference(object["tags"], .array([]))
    expectNoDifference(object["metadata"], .object([:]))
    expectNoDifference(object["ignoredOptional"], nil)
    expectNoDifference(object["ignoredDefault"], nil)
  }

  @Test
  func `Optional Nil Round Trips Through EdgeToolsValue`() throws {
    let value: EdgeToolsValue = [
      "name": "Blob",
      "display_age": 1,
      "tags": [],
      "metadata": [:],
      "address": ["city": "X"]
    ]
    let user = try MacroEncodingUser(edgeToolsValue: value)

    let decoded = try MacroEncodingUser(edgeToolsValue: user.edgeToolsValue)
    expectNoDifference(decoded.nickname, nil)
    expectNoDifference(decoded.ignoredOptional, nil)
    expectNoDifference(decoded.ignoredDefault, "default")
  }
}

@Suite
struct `EdgeToolsGenerableInitialization tests` {
  @Test
  func `Initializes Data From UTF8 String`() throws {
    expectNoDifference(try Data(edgeToolsValue: "blob"), Data("blob".utf8))
  }

  @Test
  func `Initializes Decimal`() throws {
    expectNoDifference(try Decimal(edgeToolsValue: 1.5), Decimal(1.5))
    expectNoDifference(try Decimal(edgeToolsValue: 1), Decimal(1))
    expectNoDifference(
      try Decimal(edgeToolsValue: "12345.67890123456789"),
      Decimal(string: "12345.67890123456789")
    )
  }

  @Test
  func `Initializes Optional`() throws {
    expectNoDifference(try String?(edgeToolsValue: .null), nil)
    expectNoDifference(try String?(edgeToolsValue: "blob"), "blob")
  }

  @Test
  func `Throws Type Error For Invalid Primitive Value`() {
    #expect(throws: EdgeToolsValueTypeError(expected: .string, received: .integer)) {
      try String(edgeToolsValue: 1)
    }
    #expect(throws: EdgeToolsValueTypeError(expected: .boolean, received: .string)) {
      try Bool(edgeToolsValue: "blob")
    }
    #expect(throws: EdgeToolsValueTypeError(expected: .number, received: .string)) {
      try Double(edgeToolsValue: "blob")
    }
    #expect(throws: EdgeToolsValueTypeError(expected: .integer, received: .number)) {
      try Int(edgeToolsValue: 1.5)
    }
    #expect(throws: EdgeToolsValueTypeError(expected: .array, received: .object)) {
      try [String](edgeToolsValue: [:])
    }
    #expect(throws: EdgeToolsValueTypeError(expected: .object, received: .array)) {
      try [String: String](edgeToolsValue: [])
    }
  }

  @Test
  func `Throws Type Error For Integer Overflow And Underflow`() {
    #expect(throws: EdgeToolsValueTypeError(expected: .integer, received: .integer)) {
      try Int8(edgeToolsValue: 999)
    }
    #expect(throws: EdgeToolsValueTypeError(expected: .integer, received: .integer)) {
      try UInt(edgeToolsValue: -1)
    }
  }

  @Test
  func `Initializes Macro Generated Type`() throws {
    let value: EdgeToolsValue = [
      "name": "Blob",
      "display_age": 42,
      "nickname": .null,
      "tags": ["swift", "tools"],
      "metadata": ["role": "admin"],
      "address": ["city": "Brooklyn"]
    ]

    let user = try MacroUser(edgeToolsValue: value)

    expectNoDifference(user.name, "Blob")
    expectNoDifference(user.age, 42)
    expectNoDifference(user.nickname, nil)
    expectNoDifference(user.tags, ["swift", "tools"])
    expectNoDifference(user.metadata, ["role": "admin"])
    expectNoDifference(user.address.city, "Brooklyn")
    expectNoDifference(user.ignoredOptional, nil)
    expectNoDifference(user.ignoredDefault, "default")
  }
}

@Suite
struct `SchemaComposition tests` {
  @Test(
    arguments: [
      (EdgeToolsValue.number(11.1), "11.1"),
      (1, "1"),
      (.string("blob"), "\"blob\""),
      (.boolean(true), "true"),
      (.null, "null"),
      (.array([.string("blob"), .number(11.1)]), "[\"blob\",11.1]"),
      (.array([]), "[]"),
      (.object([:]), "{}"),
      (.object(["key": .string("value")]), "{\"key\":\"value\"}")
    ]
  )
  func `Schema Value JSON`(value: EdgeToolsValue, json: String) throws {
    let data = try Self.jsonEncoder.encode(value)
    expectNoDifference(String(decoding: data, as: UTF8.self), json)

    let decodedValue = try JSONDecoder().decode(EdgeToolsValue.self, from: data)
    expectNoDifference(value, decodedValue)
  }

  @Test
  func `Ordered Key JSON Escapes Strings And Preserves Nonfinite Number Behavior`() {
    let value: EdgeToolsValue = .object([
      "escaped": .string("null\u{0}byte"),
      "nonfinite": .number(.infinity)
    ])

    expectNoDifference(
      OrderedKeyJSONWriter.encode(value),
      #"{"escaped":"null\u0000byte","nonfinite":null}"#
    )
  }

  @Test
  func `Boolean Schema JSON`() throws {
    let schema = EdgeToolsGenerationSchema.boolean(true)
    let data = try Self.jsonEncoder.encode(schema)

    expectNoDifference(String(decoding: data, as: UTF8.self), "true")
    expectNoDifference(
      try JSONDecoder().decode(EdgeToolsGenerationSchema.self, from: data),
      schema
    )
  }

  @Test
  func `Composed Schema Preserves Object Key Order`() {
    let schema = EdgeToolsGenerationSchema(
      .type(.object),
      .title("Blob"),
      .description("A blob."),
      .properties([
        "name": EdgeToolsGenerationSchema(
          .string,
          .minLength(1)
        ),
        "age": EdgeToolsGenerationSchema(
          .integer,
          .minimum(0)
        )
      ]),
      .required(["name", "age"]),
      .additionalProperties(false)
    )

    guard case .object(let object) = schema else {
      Issue.record("Expected object schema.")
      return
    }
    expectNoDifference(
      Array(object.keys),
      [.type, .title, .description, .properties, .required, .additionalProperties]
    )
  }

  @Test
  func `Later Fragments Win During Merge`() throws {
    let schema = EdgeToolsGenerationSchema(
      .type(.string),
      .minLength(1),
      .minLength(10),
      .description("first"),
      .description("second")
    )

    expectNoDifference(
      schema,
      EdgeToolsGenerationSchema([
        .type: "string",
        .minLength: 10,
        .description: "second"
      ])
    )
  }

  @Test
  func `Nullable Adds Null To Type`() throws {
    let schema = EdgeToolsGenerationSchema(
      .string,
      .minLength(1)
    )
    .nullable()

    let data = try JSONEncoder().encode(schema)
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])

    expectNoDifference(json["type"] as? [String], ["string", "null"])
    expectNoDifference(json["minLength"] as? Int, 1)
  }

  @Test
  func `Union Type Stores Ordered Type Members`() {
    let schema = EdgeToolsGenerationSchema(
      .type([.string, .boolean]),
      .minLength(10)
    )

    guard case .object(let object) = schema,
      case .array(let types)? = object[.type]
    else {
      Issue.record("Expected union type array.")
      return
    }

    expectNoDifference(types, [.string("string"), .string("boolean")])
  }

  @Test
  func `Empty Half Open Length Range Uses Its Lower Bound`() {
    let schema = EdgeToolsGenerationSchema.lengthRange(3..<3)

    expectNoDifference(
      schema,
      EdgeToolsGenerationSchema(.minLength(3), .maxLength(3))
    )
  }

  @Test
  func `Array And Object Helpers Compose`() {
    let schema = EdgeToolsGenerationSchema(
      .type(.array),
      .items(
        EdgeToolsGenerationSchema(
          .type(.object),
          .properties([
            "id": .integer,
            "name": .string
          ]),
          .required(["id", "name"])
        )
      ),
      .minItems(1),
      .uniqueItems()
    )

    guard case .object(let object) = schema else {
      Issue.record("Expected object schema.")
      return
    }
    expectNoDifference(Array(object.keys), [.type, .items, .minItems, .uniqueItems])
  }

  private static let jsonEncoder = {
    let encoder = JSONEncoder()
    return encoder
  }()
}

@EdgeToolsGenerable
private struct CustomExtractionName {
  static let extractionToolDefinition = EdgeToolDefinition(
    name: "receipt",
    description: "Extract a receipt.",
    arguments: Self.edgeToolsGenerationSchema,
    includesSchemaInInstructions: false
  )

  var value: String
}

@EdgeToolsGenerable
private struct MacroEncodingUser: Equatable {
  var name: String
  @EdgeToolsGuide(key: "display_age")
  var displayAge: Int
  var nickname: String?
  var tags: [String]
  var metadata: [String: String]
  var address: MacroEncodingAddress
  @EdgeToolsIgnored
  var ignoredOptional: String?
  @EdgeToolsIgnored
  var ignoredDefault: String = "default"
}

@EdgeToolsGenerable
private struct MacroEncodingAddress: Equatable {
  var city: String
}

@EdgeToolsGenerable
private enum MacroEncodingAction: Equatable {
  case store(MacroEncodingAddress)
  case move(Double, Double)
  case search(query: String, limit: Int?)
  case replace(String, with: String)
}

@EdgeToolsGenerable
private struct EscapedGuideKey: Equatable {
  @EdgeToolsGuide(key: "line\nbreak")
  var value: String
}

@EdgeToolsGenerable
private struct MacroUser: Equatable {
  var name: String
  @EdgeToolsGuide(key: "display_age")
  var age: Int
  var nickname: String?
  var tags: [String]
  var metadata: [String: String]
  var address: Address
  @EdgeToolsIgnored
  var ignoredOptional: String?
  @EdgeToolsIgnored
  var ignoredDefault: String = "default"
}

@EdgeToolsGenerable
private struct Address: Equatable {
  var city: String
}

private func enumCaseSchema(
  _ name: String,
  properties: KeyValuePairs<String, EdgeToolsGenerationSchema>
) -> EdgeToolsGenerationSchema {
  EdgeToolsGenerationSchema(
    .type(.object),
    .properties([
      name: EdgeToolsGenerationSchema(
        .type(.object),
        .properties(properties),
        .required(properties.map(\.0)),
        .additionalProperties(false)
      )
    ]),
    .required([name]),
    .additionalProperties(false)
  )
}
