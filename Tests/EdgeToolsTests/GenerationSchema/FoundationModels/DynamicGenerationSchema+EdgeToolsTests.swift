#if FoundationModels && canImport(FoundationModels)
  import Foundation
  import FoundationModels
  import EdgeTools
  import Testing

  @Suite
  struct `DynamicGenerationSchemaEdgeTools tests` {
    struct TestCase: Hashable, Sendable {
      let name: String
      let schema: EdgeToolsGenerationSchema
      let expectedFragments: [String]
      let isSupported: Bool
    }

    @Test(arguments: [
      TestCase(
        name: "StringValue",
        schema: EdgeToolsGenerationSchema(.string, .enum(["a", "b"])),
        expectedFragments: [#""enum":["a","b"]"#],
        isSupported: true
      ),
      TestCase(
        name: "UnsupportedStringValue",
        schema: EdgeToolsGenerationSchema(.string, .enum(["a", "b"]), .lengthRange(0...1)),
        expectedFragments: [],
        isSupported: false
      ),
      TestCase(
        name: "IntegerValue",
        schema: EdgeToolsGenerationSchema(.integer, .range(1...4)),
        expectedFragments: [#""minimum":1"#, #""maximum":4"#],
        isSupported: true
      ),
      TestCase(
        name: "NumberValue",
        schema: EdgeToolsGenerationSchema(.number, .range(1.5...4.5)),
        expectedFragments: [#""minimum":1.5"#, #""maximum":4.5"#],
        isSupported: true
      ),
      TestCase(
        name: "BooleanValue",
        schema: EdgeToolsGenerationSchema(.type(.boolean)),
        expectedFragments: [#""type":"boolean""#],
        isSupported: true
      ),
      TestCase(
        name: "NullValue",
        schema: .null,
        expectedFragments: [#""type":"null""#],
        isSupported: true
      ),
      TestCase(
        name: "ArrayValue",
        schema: EdgeToolsGenerationSchema(
          .type(.array),
          .items(.string),
          .minItems(1),
          .maxItems(3)
        ),
        expectedFragments: [#""minItems":1"#, #""maxItems":3"#],
        isSupported: true
      ),
      TestCase(
        name: "ObjectValue",
        schema: EdgeToolsGenerationSchema(
          .type(.object),
          .properties(["name": .string, "count": .integer]),
          .required(["name"])
        ),
        expectedFragments: [#""required":["name"]"#, #""x-order":["name","count"]"#],
        isSupported: true
      ),
      TestCase(
        name: "Unsupported",
        schema: true,
        expectedFragments: [],
        isSupported: false
      )
    ])
    @available(iOS 26.4, macOS 26.4, watchOS 27.0, tvOS 26.4, visionOS 26.4, *)
    func `Converts Dynamic Generation Schema`(_ testCase: TestCase) throws {
      if testCase.isSupported {
        let dynamicSchema = try DynamicGenerationSchema(
          edgeToolsGenerationSchema: testCase.schema,
          name: testCase.name
        )
        let schema = try GenerationSchema(root: dynamicSchema, dependencies: [])
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(schema)
        let json = try #require(String(data: data, encoding: .utf8))

        for fragment in testCase.expectedFragments {
          #expect(json.contains(fragment))
        }
      } else {
        #expect(throws: EdgeToolsFMConversionError.self) {
          try DynamicGenerationSchema(
            edgeToolsGenerationSchema: testCase.schema,
            name: testCase.name
          )
        }
      }
    }
  }
#endif
