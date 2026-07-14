#if FoundationModels && canImport(FoundationModels)
  import CustomDump
  import FoundationModels
  import EdgeTools
  import Testing

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
      #expect(throws: EdgeToolsFMConversionError.self) {
        try GeneratedContent(edgeToolsValue: .number(.infinity))
      }
    }
  }
#endif
