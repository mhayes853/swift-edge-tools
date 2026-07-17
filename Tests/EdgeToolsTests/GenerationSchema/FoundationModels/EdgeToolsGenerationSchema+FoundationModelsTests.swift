#if FoundationModels && canImport(FoundationModels)
  import CustomDump
  import Foundation
  import FoundationModels
  import EdgeTools
  import Testing

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
#endif
