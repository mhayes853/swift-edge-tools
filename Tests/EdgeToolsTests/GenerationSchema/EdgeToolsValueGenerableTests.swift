import CustomDump
import EdgeTools
import Foundation
import Testing

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
      .null,
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
