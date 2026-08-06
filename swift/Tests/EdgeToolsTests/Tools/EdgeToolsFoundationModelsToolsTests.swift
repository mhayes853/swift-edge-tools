#if FoundationModels && canImport(FoundationModels)
  import CustomDump
  import EdgeTools
  import Foundation
  import FoundationModels
  import Testing

  @Suite
  struct `EdgeToolsFoundationModelsTools tests` {
    @Suite
    struct `EdgeToolsFMTool tests` {
      @Test
      @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
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
      @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
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

      @Test
      @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
      func `Invokes Edge Tool From FoundationModels Input`() async throws {
        let tool = try FMEdgeTool(EdgeWeatherTool())
        let arguments = try FMEdgeToolArguments<EdgeWeatherInput>(
          GeneratedContent(properties: ["city": "Brooklyn"])
        )
        let output = try await tool.call(arguments: arguments)

        expectNoDifference(tool.name, "edgeWeather")
        expectNoDifference(tool.description, "Returns EdgeTools weather.")
        expectNoDifference(output, "Cloudy in Brooklyn")
      }
    }
  }

  @Generable(description: "Weather query arguments")
  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  private struct WeatherArgs: Equatable {
    var city: String
    var units: String?
  }

  @available(iOS 26.0, macOS 26.0, watchOS 27.0, tvOS 26.0, visionOS 26.0, *)
  private struct WeatherTool: Tool {
    typealias Output = String

    let name = "getWeather"
    let description = "Returns the current weather for a city."

    func call(arguments: WeatherArgs) async throws -> String {
      "Sunny in \(arguments.city)\(arguments.units.map { " (\($0))" } ?? "")"
    }
  }

  @EdgeToolsGenerable(.title("EdgeWeatherInput"))
  private struct EdgeWeatherInput: Sendable {
    let city: String
  }

  private struct EdgeWeatherTool: EdgeTool {
    let name = "edgeWeather"
    let description = "Returns EdgeTools weather."

    func invoke(input: EdgeWeatherInput) async throws -> String {
      "Cloudy in \(input.city)"
    }
  }
#endif
