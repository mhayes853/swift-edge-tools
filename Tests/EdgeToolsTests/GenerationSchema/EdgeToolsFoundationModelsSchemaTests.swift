#if FoundationModels && canImport(FoundationModels)
  import CustomDump
  import EdgeTools
  import Foundation
  import FoundationModels
  import Testing

  @Suite
  struct EdgeToolsFoundationModelsSchemaConsolidatedTests {
    @Suite("EdgeToolsValue FoundationModels tests")
    struct EdgeToolsValueFMTests {
      @Test
      @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
      func `Converts Generated Content In Both Directions`() throws {
        let content = GeneratedContent(
          properties: [
            "name": "Blob",
            "count": 2,
            "score": 2.5,
            "enabled": true,
            "items": GeneratedContent(elements: ["first", "second"]),
            "missing": Optional<String>.none
          ]
        )

        let value = try EdgeToolsValue(generatedContent: content)
        let convertedContent = try GeneratedContent(edgeToolsValue: value)

        expectNoDifference(convertedContent, content)
      }

      @Test
      @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
      func `Rejects Nonfinite Numbers`() {
        let error = #expect(throws: EdgeToolsFMError.self) {
          try GeneratedContent(edgeToolsValue: .number(.infinity))
        }
        expectNoDifference(error?.code, .nonFiniteNumber)
      }
    }

    @Suite("EdgeToolsGenerationSchema FoundationModels tests")
    struct EdgeToolsGenerationSchemaFMTests {
      @Test
      @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
      func `Converts Generation Schema In Both Directions`() throws {
        let schema = EdgeToolsGenerationSchema(
          .type(.object),
          .title("WeatherArgs"),
          .description("Weather query arguments"),
          .properties([
            "city": EdgeToolsGenerationSchema(.string, .enum(["Brooklyn", "Cupertino"])),
            "units": .string.nullable()
          ]),
          .required(["city"]),
          .additionalProperties(false)
        )

        let generationSchema = try GenerationSchema(edgeToolsGenerationSchema: schema)
        let convertedSchema = try EdgeToolsGenerationSchema(generationSchema: generationSchema)
        let expectedSchema = EdgeToolsGenerationSchema(
          .type(.object),
          .title("WeatherArgs"),
          .description("Weather query arguments"),
          .properties([
            "city": EdgeToolsGenerationSchema(.string, .enum(["Brooklyn", "Cupertino"])),
            "units": .string
          ]),
          .required(["city"]),
          .additionalProperties(false),
          .xOrder(["city", "units"])
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let convertedData = try encoder.encode(convertedSchema)
        let expectedData = try encoder.encode(expectedSchema)
        let convertedJSON = try #require(String(data: convertedData, encoding: .utf8))
        let expectedJSON = try #require(String(data: expectedData, encoding: .utf8))

        expectNoDifference(convertedJSON, expectedJSON)
      }
    }

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
          #expect(throws: EdgeToolsFMError.self) {
            try DynamicGenerationSchema(
              edgeToolsGenerationSchema: testCase.schema,
              name: testCase.name
            )
          }
        }
      }
    }
  }
#endif
