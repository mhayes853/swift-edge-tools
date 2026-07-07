import CustomDump
import Foundation
import Needle
import Testing

@Suite
struct `NeedleGenerationSchema tests` {
  @Test(
    arguments: [
      (NeedleValue.number(11.1), "11.1"),
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
  func `Schema Value JSON`(value: NeedleValue, json: String) throws {
    let data = try Self.jsonEncoder.encode(value)
    expectNoDifference(String(decoding: data, as: UTF8.self), json)

    let decodedValue = try JSONDecoder().decode(NeedleValue.self, from: data)
    expectNoDifference(value, decodedValue)
  }

  @Test
  func `Needle Value Object Preserves Key Ordering`() {
    let value: NeedleValue = .object([
      "b": 2,
      "a": 1,
      "c": 3,
    ])

    guard case .object(let object) = value else {
      Issue.record("Expected object value.")
      return
    }
    expectNoDifference(Array(object.keys), ["b", "a", "c"])
  }

  @Test
  func `Boolean Schema JSON`() throws {
    let schema = NeedleGenerationSchema.boolean(true)
    let data = try Self.jsonEncoder.encode(schema)

    expectNoDifference(String(decoding: data, as: UTF8.self), "true")
    expectNoDifference(try JSONDecoder().decode(NeedleGenerationSchema.self, from: data), schema)
  }

  @Test
  func `Composed Schema Preserves Object Key Order`() {
    let schema = NeedleGenerationSchema(
      .type(.object),
      .title("Blob"),
      .description("A blob."),
      .properties([
        "name": NeedleGenerationSchema(
          .string,
          .minLength(1)
        ),
        "age": NeedleGenerationSchema(
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
    expectNoDifference(Array(object.keys), [.type, .title, .description, .properties, .required, .additionalProperties])
  }

  @Test
  func `Later Fragments Win During Merge`() throws {
    let schema = NeedleGenerationSchema(
      .type(.string),
      .minLength(1),
      .minLength(10),
      .description("first"),
      .description("second")
    )

    expectNoDifference(
      schema,
      NeedleGenerationSchema([
        .type: "string",
        .minLength: 10,
        .description: "second"
      ])
    )
  }

  @Test
  func `Nullable Adds Null To Type`() throws {
    let schema = NeedleGenerationSchema(
      .string,
      .minLength(1)
    ).nullable()

    let data = try JSONEncoder().encode(schema)
    expectNoDifference(
      String(decoding: data, as: UTF8.self),
      #"{"type":["string","null"],"minLength":1}"#
    )
  }

  @Test
  func `Union Type Stores Ordered Type Members`() {
    let schema = NeedleGenerationSchema(
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
    let schema = NeedleGenerationSchema(
      .type(.array),
      .items(
        NeedleGenerationSchema(
          .type(.object),
          .properties([
            "id": .integer,
            "name": .string,
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
