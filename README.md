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
let model = try NeedleCoreAI(from: modelURL) // WWDC Rumors

let toolDefinition = NeedleToolDefinition(
  name: "get_weather",
  arguments: toolSchema
)

let response = try model.invoke(messages: [], tools: [toolDefinition])
print(response.messages, response.toolCalls)

struct GetWeather: NeedleTool {
  @NeeldeGenerable
  struct Input {
    let city: String
  }

  func invoke(input: sending Input) async throws -> String {
    // ...
  }
}

// Can also construct with NeedleCactus, NeedleONNX, etc.
let session = try NeedleSession<NeedleMLX>(from: modelURL)

try await session.prefill(prompt: "What is the weather in ", tools: [GetWeather()])

let response = try await session.invoke(
  prompt: "What is the weather in San Francisco?", 
  tools: [GetWeather()]
)
print(response) // [String]

let stream = try await session.stream(prompt: "What is the weather in San Francisco?", tools: [GetWeather()])
for try await token in stream {
  print(token.stringValue, token.id)
}

struct GetPopulation: NeedleTool {
  @NeedleGenerable
  struct Input {
    let city: String
  }

  func invoke(input: sending Input) async throws -> Int {
    // ...
  }
}

// Dynamic
let session = try NeedleSession<NeedleMLX>(from: modelURL)
let response = try await session.invoke(
  prompt: "What is the weather in San Francisco?", 
  tools: [GetWeather(), GetPopulation()]
)
print(response) // NeedleSession.DynamicToolInvocations

let weatherResult = response[0].result(of: GetWeather.self) // Result<NeedleToolInvocation<GetWeather>, any Error>?
let populationResult = response[0].output(of: GetPopulation.self) // Result<NeedleToolInvocation<GetPopulation>, any Error>?

// Static

@NeedleStaticToolCollection
enum SessionTools {
  case getWeather(GetWeather)
  case getPopulation(GetPopulation)
}

// Macro Generates
// @NeedleStaticToolCollectionCase(GetWeather(/* Custom args */)) on case if one wants custom configuration
extension SessionTools {
  static let tools: [any NeedleTool] = [GetWeather(), GetPopulation()]

  enum Output: NeedleStaticToolsCollectionOutput {
    case getWeather(GetWeather.Output)
    case getPopulation(GetPopulation.Output)

    // Generates getter func body...
  }
}

let session = try NeedleSession<NeedleMLX>(from: modelURL)
let response = try await session.invoke(
  prompt: "What is the weather in San Francisco?", 
  tools: SessionTools.self
)
print(response) // NeedleSession.StaticToolInvocations<SessionTools.Output>
```
