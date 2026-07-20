import CustomDump
import EdgeTools
import Foundation
import Testing

@Suite
struct EdgeToolsGenerationSchemaConsolidatedTests {
  @Suite
  struct `EdgeToolsEncoding tests` {
    @Test
    func `String Encodes To String Value`() {
      expectNoDifference("blob".edgeToolsValue, .string("blob"))
    }

    @Test
    func `Bool Encodes To Boolean Value`() {
      expectNoDifference(true.edgeToolsValue, .boolean(true))
      expectNoDifference(false.edgeToolsValue, .boolean(false))
    }

    @Test
    func `Signed Integers Encode To Integer Value`() {
      expectNoDifference((-1).edgeToolsValue, .integer(-1))
      expectNoDifference(Int8(2).edgeToolsValue, .integer(2))
      expectNoDifference(Int16(3).edgeToolsValue, .integer(3))
      expectNoDifference(Int32(4).edgeToolsValue, .integer(4))
      expectNoDifference(Int64(5).edgeToolsValue, .integer(5))
      expectNoDifference(Int64.max.edgeToolsValue, .integer(Int(Int64.max)))
    }

    @Test
    func `Unsigned Integers Encode To Integer Value`() {
      expectNoDifference(UInt(2).edgeToolsValue, .integer(2))
      expectNoDifference(UInt8(3).edgeToolsValue, .integer(3))
      expectNoDifference(UInt16(4).edgeToolsValue, .integer(4))
      expectNoDifference(UInt32(5).edgeToolsValue, .integer(5))
      expectNoDifference(UInt64(6).edgeToolsValue, .integer(6))
      expectNoDifference(UInt64(Int.max).edgeToolsValue, .integer(Int.max))
    }

    @Test
    func `Doubles And Floats Encode To Number Value`() {
      expectNoDifference((1.5).edgeToolsValue, .number(1.5))
      expectNoDifference(Float(2.5).edgeToolsValue, .number(2.5))
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
    func `Array Encodes To Array Value`() {
      expectNoDifference([Int]().edgeToolsValue, .array([]))
      expectNoDifference(
        [1, 2, 3].edgeToolsValue,
        .array([.integer(1), .integer(2), .integer(3)])
      )
      expectNoDifference(["a", "b"].edgeToolsValue, .array([.string("a"), .string("b")]))
    }

    @Test
    func `Dictionary Encodes To Object Value`() {
      expectNoDifference([String: Int]().edgeToolsValue, .object([:]))

      let encoded = ["one": 1, "two": 2].edgeToolsValue
      guard case .object(let object) = encoded else {
        Issue.record("Expected dictionary to encode as an object value.")
        return
      }
      let expected = ["one": EdgeToolsValue.integer(1), "two": .integer(2)]
      expectNoDifference(
        Dictionary(uniqueKeysWithValues: object.map { ($0.key, $0.value) }),
        expected
      )
    }

    @Test
    func `EdgeToolsValue Encodes To Itself`() {
      expectNoDifference(EdgeToolsValue.null.edgeToolsValue, .null)
      expectNoDifference(EdgeToolsValue.string("blob").edgeToolsValue, .string("blob"))
    }

    @Test
    func `Macro Generated Type Round Trips Through EdgeToolsValue`() throws {
      let value: EdgeToolsValue = [
        "name": "Blob",
        "display_age": 42,
        "nickname": "blob",
        "tags": ["swift", "needle"],
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
  struct `EdgeToolsGenerable initialization tests` {
    @Test
    func `Initializes String`() throws {
      expectNoDifference(try String(edgeToolsValue: "blob"), "blob")
    }

    @Test
    func `Initializes Bool`() throws {
      expectNoDifference(try Bool(edgeToolsValue: true), true)
    }

    @Test
    func `Initializes Double`() throws {
      expectNoDifference(try Double(edgeToolsValue: 1.5), 1.5)
      expectNoDifference(try Double(edgeToolsValue: 1), 1)
    }

    @Test
    func `Initializes Float`() throws {
      expectNoDifference(try Float(edgeToolsValue: 1.5), 1.5)
      expectNoDifference(try Float(edgeToolsValue: 1), 1)
    }

    @Test
    func `Initializes Signed And Unsigned Integers`() throws {
      expectNoDifference(try Int8(edgeToolsValue: 1), 1)
      expectNoDifference(try UInt(edgeToolsValue: 1), 1)
    }

    @Test
    func `Initializes Data From UTF8 String`() throws {
      expectNoDifference(try Data(edgeToolsValue: "blob"), Data("blob".utf8))
    }

    @Test
    func `Initializes Decimal`() throws {
      expectNoDifference(try Decimal(edgeToolsValue: 1.5), Decimal(1.5))
      expectNoDifference(try Decimal(edgeToolsValue: 1), Decimal(1))
    }

    @Test
    func `Initializes Optional`() throws {
      expectNoDifference(try String?(edgeToolsValue: .null), nil)
      expectNoDifference(try String?(edgeToolsValue: "blob"), "blob")
    }

    @Test
    func `Initializes Array`() throws {
      expectNoDifference(try [Int](edgeToolsValue: [1, 2, 3]), [1, 2, 3])
    }

    @Test
    func `Initializes Dictionary`() throws {
      expectNoDifference(
        try [String: Int](edgeToolsValue: ["one": 1, "two": 2]),
        ["one": 1, "two": 2]
      )
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
        "tags": ["swift", "needle"],
        "metadata": ["role": "admin"],
        "address": ["city": "Brooklyn"]
      ]

      let user = try MacroUser(edgeToolsValue: value)

      expectNoDifference(user.name, "Blob")
      expectNoDifference(user.age, 42)
      expectNoDifference(user.nickname, nil)
      expectNoDifference(user.tags, ["swift", "needle"])
      expectNoDifference(user.metadata, ["role": "admin"])
      expectNoDifference(user.address.city, "Brooklyn")
      expectNoDifference(user.ignoredOptional, nil)
      expectNoDifference(user.ignoredDefault, "default")
    }
  }

  @Suite
  struct `EdgeToolsGenerationSchema tests` {
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
    func `Edge Tools Value Object Preserves Key Ordering`() {
      let value: EdgeToolsValue = .object([
        "b": 2,
        "a": 1,
        "c": 3
      ])

      guard case .object(let object) = value else {
        Issue.record("Expected object value.")
        return
      }
      expectNoDifference(Array(object.keys), ["b", "a", "c"])
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

  @Suite
  struct `EdgeToolsValue EdgeToolsGenerable tests` {
    @Test
    func `EdgeToolsValue Universal Schema Accepts Every Value Type`() throws {
      let schema = EdgeToolsValue.edgeToolsGenerationSchema

      let values: [EdgeToolsValue] = [
        .string("blob"),
        .boolean(true),
        .boolean(false),
        .number(3.14),
        .integer(42),
        .array([.string("blob"), .integer(1)]),
        .object(["key": .string("value")]),
        .null
      ]

      for value in values {
        let data = try JSONEncoder().encode(schema)
        let decodedSchema = try JSONDecoder().decode(EdgeToolsGenerationSchema.self, from: data)

        expectNoDifference(decodedSchema, schema)
        expectNoDifference(String(decoding: data, as: UTF8.self), "true")
        expectNoDifference(EdgeToolsValue(edgeToolsValue: value), value)
      }
    }
  }

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
