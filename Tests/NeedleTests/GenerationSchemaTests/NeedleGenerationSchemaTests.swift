import CustomDump
import Foundation
import Needle
import SnapshotTesting
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

  @Test(
    arguments: [
      (
        NeedleGenerationSchema.ValueSchema.Array.array(
          items: NeedleGenerationSchema.object(valueSchema: .null)
        ),
        #"{"items":{"type":"null"}}"#
      ),
      (
        NeedleGenerationSchema.ValueSchema.Array.array(
          prefixItems: [
            NeedleGenerationSchema.object(valueSchema: .null),
            NeedleGenerationSchema.object(valueSchema: .boolean),
          ]
        ),
        #"{"prefixItems":[{"type":"null"},{"type":"boolean"}]}"#
      )
    ]
  )
  func `Array Items And Prefix Items JSON`(
    array: NeedleGenerationSchema.ValueSchema.Array, json: String
  ) throws {
    let data = try Self.jsonEncoder.encode(array)
    expectNoDifference(String(decoding: data, as: UTF8.self), json)

    let decodedValue = try JSONDecoder()
      .decode(NeedleGenerationSchema.ValueSchema.Array.self, from: data)
    expectNoDifference(array, decodedValue)
  }

  @Test(
    arguments: [
      (
        NeedleGenerationSchema.ValueSchema.Array.array(
          contains: NeedleGenerationSchema.object(valueSchema: .number()),
          minContains: 1,
          maxContains: 3
        ),
        #"{"contains":{"type":"number"},"maxContains":3,"minContains":1}"#
      )
    ]
  )
  func `Array Min And Max Contains JSON`(
    array: NeedleGenerationSchema.ValueSchema.Array, json: String
  ) throws {
    let data = try Self.jsonEncoder.encode(array)
    expectNoDifference(String(decoding: data, as: UTF8.self), json)

    let decodedValue = try JSONDecoder()
      .decode(NeedleGenerationSchema.ValueSchema.Array.self, from: data)
    expectNoDifference(array, decodedValue)
  }

  @Test(
    arguments: [
      (
        NeedleGenerationSchema.ValueSchema.Object.object(
          properties: ["name": .string()],
          dependentRequired: ["name": ["age"]]
        ),
        #"{"dependentRequired":{"name":["age"]},"properties":{"name":{"type":"string"}}}"#
      )
    ]
  )
  func `Object Dependent Required JSON`(
    object: NeedleGenerationSchema.ValueSchema.Object, json: String
  ) throws {
    let data = try Self.jsonEncoder.encode(object)
    expectNoDifference(String(decoding: data, as: UTF8.self), json)

    let decodedValue = try JSONDecoder()
      .decode(NeedleGenerationSchema.ValueSchema.Object.self, from: data)
    expectNoDifference(object, decodedValue)
  }

  @Test
  func `Union Type JSON`() throws {
    let value = NeedleGenerationSchema.union(string: .string(minLength: 10), bool: true)
    let json = "{\"minLength\":10,\"type\":[\"string\",\"boolean\"]}"

    let data = try Self.jsonEncoder.encode(value)
    expectNoDifference(String(decoding: data, as: UTF8.self), json)

    let decodedValue = try JSONDecoder().decode(NeedleGenerationSchema.self, from: data)
    expectNoDifference(value, decodedValue)
  }

  @Test
  func `Boolean Schema JSON`() throws {
    let schema = NeedleGenerationSchema.boolean(true)
    let json = "true"

    let data = try Self.jsonEncoder.encode(schema)
    expectNoDifference(String(decoding: data, as: UTF8.self), json)

    let decodedValue = try JSONDecoder().decode(NeedleGenerationSchema.self, from: data)
    expectNoDifference(schema, decodedValue)
  }

  @Test
  func `Bool Value Type Schema JSON`() throws {
    let schema = NeedleGenerationSchema.bool()
    let json = "{\"type\":\"boolean\"}"

    let data = try Self.jsonEncoder.encode(schema)
    expectNoDifference(String(decoding: data, as: UTF8.self), json)

    let decodedValue = try JSONDecoder().decode(NeedleGenerationSchema.self, from: data)
    expectNoDifference(schema, decodedValue)
  }

  @Test
  func `Empty Schema JSON`() throws {
    let schema = NeedleGenerationSchema.object(valueSchema: nil)
    let json = "{}"

    let data = try Self.jsonEncoder.encode(schema)
    expectNoDifference(String(decoding: data, as: UTF8.self), json)

    let decodedValue = try JSONDecoder().decode(NeedleGenerationSchema.self, from: data)
    expectNoDifference(schema, decodedValue)
  }

  @Test(
    .serialized,
    arguments: [
      NeedleGenerationSchema.Object(
        title: "blob",
        description: "A mysterious loreful character.",
        valueSchema: .object(properties: ["name": .object(valueSchema: .string())])
      ),
      NeedleGenerationSchema.Object(
        title: "n",
        description: "A number.",
        valueSchema: .number(minimum: 10.1, maximum: 20.2)
      ),
      NeedleGenerationSchema.Object(title: "b", description: "A boolean.", valueSchema: .boolean),
      NeedleGenerationSchema.Object(
        title: "Nullable",
        description: "A nullable property.",
        valueSchema: .null
      ),
      NeedleGenerationSchema.Object(
        title: "Array",
        description: "An array",
        valueSchema: .array(
          items: .object(valueSchema: .string()),
          minItems: 10,
          uniqueItems: true
        )
      ),
      NeedleGenerationSchema.Object(
        title: "Enum",
        valueSchema: nil,
        enum: [.boolean(true), .string("blob")]
      ),
      NeedleGenerationSchema.Object(
        title: "Union",
        valueSchema: .union(string: .string(), isBoolean: true)
      ),
      NeedleGenerationSchema.Object(
        title: "Integer",
        description: "An integer",
        valueSchema: .integer(minimum: 10, maximum: 20)
      )
    ]
  )
  func `Object Schema JSON`(object: NeedleGenerationSchema.Object) throws {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
    assertSnapshot(of: NeedleGenerationSchema.object(object), as: .json(encoder))
    let decoded = try JSONDecoder()
      .decode(
        NeedleGenerationSchema.self,
        from: Self.jsonEncoder.encode(NeedleGenerationSchema.object(object))
      )
    expectNoDifference(decoded, NeedleGenerationSchema.object(object))
  }

  @Test
  func `Prioritizes Number Properties Over Integer Properties When Encoding`() throws {
    let schema = NeedleGenerationSchema.object(
      valueSchema: .union(number: .number(minimum: 10), integer: .integer(minimum: 12))
    )
    let data = try Self.jsonEncoder.encode(schema)
    let decoded = try JSONDecoder().decode(NeedleGenerationSchema.self, from: data)

    switch (schema, decoded) {
    case (.object(let schema), .object(let decoded)):
      expectNoDifference(schema.valueSchema?.number?.minimum, decoded.valueSchema?.number?.minimum)
      expectNoDifference(decoded.valueSchema?.number?.minimum, 10)
      expectNoDifference(decoded.valueSchema?.integer?.minimum, 10)
    default:
      break
    }
  }

  @Test
  func `Single Value Type When Single ValueSchema`() {
    let schema = NeedleGenerationSchema.Object(valueSchema: .number(minimum: 10))
    expectNoDifference(schema.type, .number)
  }

  @Test
  func `Union Value Type When Union ValueSchema`() {
    let schema = NeedleGenerationSchema.Object(
      valueSchema: .union(number: .number(minimum: 10), isNullable: true)
    )
    expectNoDifference(schema.type, [.number, .null])
  }

  @Test
  func `Typed Convenience Schemas Are Equivalent To Legacy APIs`() {
    expectNoDifference(
      NeedleGenerationSchema.string(
        title: "Title",
        description: "Description",
        minLength: 1,
        maxLength: 10,
        pattern: "[a-z]+",
        default: "blob",
        examples: ["blob"],
        enum: ["blob", "jr"],
        const: "blob"
      ),
      NeedleGenerationSchema.object(
        title: "Title",
        description: "Description",
        valueSchema: .string(minLength: 1, maxLength: 10, pattern: "[a-z]+"),
        default: "blob",
        examples: ["blob"],
        enum: ["blob", "jr"],
        const: "blob"
      )
    )

    expectNoDifference(
      NeedleGenerationSchema.number(title: "N", description: "D", minimum: 1.5, maximum: 10),
      NeedleGenerationSchema.object(
        title: "N",
        description: "D",
        valueSchema: .number(minimum: 1.5, maximum: 10)
      )
    )

    expectNoDifference(
      NeedleGenerationSchema.integer(title: "N", description: "D", minimum: 1, maximum: 10),
      NeedleGenerationSchema.object(
        title: "N",
        description: "D",
        valueSchema: .integer(minimum: 1, maximum: 10)
      )
    )

    expectNoDifference(
      NeedleGenerationSchema.array(
        title: "A",
        description: "D",
        minItems: 1,
        maxItems: 3,
        uniqueItems: true
      ),
      NeedleGenerationSchema.object(
        title: "A",
        description: "D",
        valueSchema: .array(minItems: 1, maxItems: 3, uniqueItems: true)
      )
    )

    expectNoDifference(
      NeedleGenerationSchema.object(
        title: "Obj",
        description: "D",
        properties: ["name": .string(minLength: 1)],
        required: ["name"]
      ),
      NeedleGenerationSchema.object(
        title: "Obj",
        description: "D",
        valueSchema: .object(
          properties: ["name": .string(minLength: 1)],
          required: ["name"]
        )
      )
    )

    expectNoDifference(
      NeedleGenerationSchema.null(title: "Null", description: "D"),
      NeedleGenerationSchema.object(title: "Null", description: "D", valueSchema: .null)
    )

    expectNoDifference(
      NeedleGenerationSchema.bool(title: "Bool", description: "D"),
      NeedleGenerationSchema.object(title: "Bool", description: "D", valueSchema: .boolean)
    )

    expectNoDifference(
      NeedleGenerationSchema.union(
        title: "Union",
        description: "D",
        string: .string(minLength: 1),
        bool: true,
        null: true
      ),
      NeedleGenerationSchema.object(
        title: "Union",
        description: "D",
        valueSchema: .union(string: .string(minLength: 1), isBoolean: true, isNullable: true)
      )
    )
  }

  @Test
  func `Merge NeedleGenerationSchema Helper Merges Metadata For Object Schemas`() {
    let base = NeedleGenerationSchema.string(minLength: 1)
    let merged = _needleMergeGenerationSchema(base, title: "Title", description: "Description")

    expectNoDifference(
      merged,
      NeedleGenerationSchema.string(title: "Title", description: "Description", minLength: 1)
    )
  }

  @Test
  func `Merge NeedleGenerationSchema Helper Leaves Boolean Schemas Unchanged`() {
    let base = NeedleGenerationSchema.boolean(true)
    let merged = _needleMergeGenerationSchema(base, title: "Ignored", description: "Ignored")
    expectNoDifference(merged, base)
  }

  private static let jsonEncoder = {
    let encoder = JSONEncoder()
    encoder.outputFormatting = .sortedKeys
    return encoder
  }()
}
