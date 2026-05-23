# Swift Needle

A runtime for [Cactus Needle](https://github.com/cactus-compute/needle) in pure Swift.

## Usage

```swift
import Needle

let modelURL = try await NeedleMLX.download(from: .cactusNeedleMLXHuggingFace) { progress in
  print("Progress", progress)
}

// Any of (start with MLX)
let model = try NeedleMLX(from: modelURL)
let model = try NeedleONNX(from: modelURL)
let model = try NeedleCactus(from: modelURL)
let model = try NeedleCoreML(from: modelURL)

let toolDefinition = NeedleToolDefinition(
  name: "get_weather",
  arguments: toolSchema
)

let response = try model.invoke(messages: [], tools: [toolDefinition])
print(response.messages, response.toolCalls)

struct GetWeather: NeedleSession.Tool {
  @JSONSchema
  struct Input {
    let city: String
  }

  func invoke(input: sending Input) async throws -> String {
    // ...
  }
}

// Can also construct with NeedleCactus, NeedleONNX, etc.
let session = try NeedleSession<NeedleMLX>(from: modelURL, tools: [GetWeather()])

try await session.prefill(prompt: "What is the weather in ")

let response = try await session.invoke(prompt: "What is the weather in San Francisco?")
print(response) // [String]

let stream = try await session.stream(prompt: "What is the weather in San Francisco?")
for try await token in stream {
  print(token.stringValue, token.id)
}

struct GetPopulation: NeedleSession.Tool {
  @JSONSchema
  struct Input {
    let city: String
  }

  func invoke(input: sending Input) async throws -> Int {
    // ...
  }
}

// Dynamic
let session = try NeedleSession(from: modelURL, tools: [GetWeather(), GetPopulation()])
let response = try await session.invoke(prompt: "What is the weather in San Francisco?")
print(response) // NeedleSession.DynamicToolInvocations

let weatherResponse = response[0].output(of: GetWeather.self) // String?
let weatherInput = response[0].input(of: GetWeather.self) // GetWeather.Input?

let populationResponse = response[0].output(of: GetPopulation.self) // Int?
let populationInput = response[0].input(of: GetPopulation.self) // GetPopulation.Input?

// Static

@NeedleStaticToolCollection
enum SessionTools {
  case getWeather(GetWeather)
  case getPopulation(GetPopulation)
}

// Macro Generates
// @NeedleStaticToolCollectionCase(GetWeather(/* Custom args */)) on case if one wants custom configuration
extension SessionTools {
  static let tools: [any NeedleSession.Tool] = [
    GetWeather(),
    GetPopulation()
  ]

  enum Output {
    case getWeather(GetWeather.Output)
    case getPopulation(GetPopulation.Output)
  }
}

let session = try NeedleSession(from: modelURL, staticCollection: SessionTools.self)
let response = try await session.invoke(prompt: "What is the weather in San Francisco?")
print(response) // NeedleSession.StaticToolInvocations<SessionTools.Output>
```
