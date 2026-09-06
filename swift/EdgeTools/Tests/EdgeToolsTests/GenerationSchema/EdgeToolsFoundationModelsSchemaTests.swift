#if FoundationModels && canImport(FoundationModels)
  import CustomDump
  import EdgeTools
  import Foundation
  import FoundationModels
  import Testing

  @Suite
  struct `EdgeToolsFoundationModelsSchema tests` {
    @Suite
    struct `EdgeToolsValueFoundationModels tests` {
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
        let error = #expect(throws: FMConversionError.self) {
          try GeneratedContent(edgeToolsValue: .number(.infinity))
        }
        expectNoDifference(error?.code, .nonFiniteNumber)
      }
    }

    @Suite
    struct `EdgeToolsGenerationSchemaFoundationModels tests` {
      @Test
      @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
      func `Converts Generation Schema From FoundationModels`() throws {
        let schema = try EdgeToolsGenerationSchema(generationSchema: RouteQuery.generationSchema)

        guard case .object(let root) = schema else {
          Issue.record("Expected object schema, got \(schema)")
          return
        }
        expectNoDifference(root[.title], .string("RouteQuery"))
        expectNoDifference(root[.description], .string("A routing decision"))
        expectNoDifference(root[.required], .array([.string("destination")]))

        guard case .object(let properties)? = root[.properties] else {
          Issue.record("Expected properties object.")
          return
        }
        expectNoDifference(properties.keys.sorted(), ["confidence", "destination"])

        guard case .object(let destination)? = properties["destination"] else {
          Issue.record("Expected destination schema object.")
          return
        }
        expectNoDifference(
          destination["enum"],
          .array([.string("onDevice"), .string("privateCloud")])
        )
      }
    }
  }

  @Generable(description: "A routing decision")
  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  private struct RouteQuery {
    @Guide(description: "Where to send the query", .anyOf(["onDevice", "privateCloud"]))
    var destination: String
    var confidence: Double?
  }
#endif
