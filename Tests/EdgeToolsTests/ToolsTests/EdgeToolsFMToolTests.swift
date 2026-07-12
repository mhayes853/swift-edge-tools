#if canImport(FoundationModels)
  import CustomDump
  import Foundation
  import FoundationModels
  import EdgeTools
  import Testing

  @Suite
  struct `EdgeToolsFMTool tests` {
    @Test
    @available(iOS 26.0, macOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    func `Contains Expected Metadata And Properties`() throws {
      let tool = EdgeToolsFMTool(WeatherTool())
      let schema = tool.arguments

      expectNoDifference(tool.name, "getWeather")
      expectNoDifference(tool.description, "Returns the current weather for a city.")

      guard case .object(let root) = schema else {
        Issue.record("Expected object schema, got \(schema)")
        return
      }
      expectNoDifference(root[.description], .string("Weather query arguments"))
      expectNoDifference(root[.title], .string("WeatherArgs"))

      guard case .object(let properties)? = root[.properties] else {
        Issue.record("Expected properties object.")
        return
      }
      expectNoDifference(properties.keys.sorted(), ["city", "units"])
      expectNoDifference(root[.required], .array([.string("city")]))
    }

    @Test
    @available(iOS 26.0, macOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    func `Preserves Edge Tools Value And Typed Arguments`() throws {
      let input = try EdgeToolsFMToolInput<WeatherArgs>(
        edgeToolsValue: ["city": "Brooklyn", "units": "metric"]
      )

      expectNoDifference(input.edgeToolsValue, ["city": "Brooklyn", "units": "metric"])
      expectNoDifference(input.arguments, WeatherArgs(city: "Brooklyn", units: "metric"))
    }

    @Test
    @available(iOS 26.0, macOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
    func `Invokes FoundationModels Tool From Wrapped Input`() async throws {
      let tool = EdgeToolsFMTool(WeatherTool())
      let input = try EdgeToolsFMToolInput<WeatherArgs>(
        edgeToolsValue: [
          "city": "Brooklyn",
          "units": "metric"
        ]
      )
      let output = try await tool.invoke(input: input)

      expectNoDifference(output, "Sunny in Brooklyn (metric)")
    }

    private static let jsonEncoder = JSONEncoder()
  }

  @Generable(description: "Weather query arguments")
  @available(iOS 26.0, macOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
  private struct WeatherArgs: Equatable {
    var city: String
    var units: String?
  }

  @available(iOS 26.0, macOS 26.0, watchOS 26.0, tvOS 26.0, visionOS 26.0, *)
  private struct WeatherTool: Tool {
    typealias Output = String

    let name = "getWeather"
    let description = "Returns the current weather for a city."

    func call(arguments: WeatherArgs) async throws -> String {
      "Sunny in \(arguments.city)\(arguments.units.map { " (\($0))" } ?? "")"
    }
  }
#endif
