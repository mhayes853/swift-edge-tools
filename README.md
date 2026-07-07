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

New JSON Schema API
```swift
let schema = NeedleGenerationSchema(
  .type(.string),
  .minLength(1)
)

// Raw init (each value is a NeedleValue)
let schema = NeedleGenerationSchema([
  "type": "string",
  "minLength": 1
])

// Composite Keys
let schema = NeedleGenerationSchema(
  .type(.string),
  .lengthRange(1...10),
  .pattern("[0-9a-zA-Z]+")
)

let schema = NeedleGenerationSchema(
  .type(.number),
  .range(1.2...10.1)
)

// Arrays
let schema = NeedleGenerationSchema(
  .array, // OR .type(.array)
  .properties(.type(.string), .minLength(1))
)

// Objects
let schema = NeedleGenerationSchema(
  .type(.object), // OR .type(.object)
  .properties([
    "name": NeedleGenerationSchema(
      .string, // OR .type(.string)
      .minLength(1)
    )
  ])
)

let schema = NeedleGenerationSchema(
  .type(.object), // OR .type(.object)
  .properties([
    "name": [
      .string, // OR .type(.string)
      .minLength(1)
    ]
  ])
)

// Unions
let schema = NeedleGenerationSchema(
  .type([.string, .number]),
  .minLength(1), // string constraint
  .minimum(1) // number constraint
)

// Metakeys
let schema = NeedleGenerationSchema(
  .type(.number),
  .range(1.2...10.1),
  .oneOf([otherSchema, ...]),
  .allOf([otherSchema, ...])
)

public enum NeedleGenerationSchema: Hashable, Sendable, Codable {
  public struct Key: RawRepresentable, ExpressibleByStringLiteral, Codable {
    // Have extensions for all common keys in JSON Schema standard ("type", "minLength", etc.)
  }

  // Ordered dictionary would guarantee ordering for Needle Prompts
  case object(OrderedDictionary<Key: NeedleValue>)
  case boolean(Bool)

  public init(_ properties: OrderedDictionary<Key: NeedleValue>)

  // Allows inits that you see above with strongly typed properties. Implementation would merge underlying property dicts
  public init(_ schemas: some Sequence<Self>)
  public init(_ schemas: Self...)
}

extension NeedleGenerationSchema: ExpressibleByArrayLiteral {}
extension NeedleGenerationSchema: ExpressibleByDictionaryLiteral {}
extension NeedleGenerationSchema: ExpressibleByBooleanLiteral {}

extension NeedleGenerationSchema {
  public static func type(_ type: ValueType) -> Self {
    Self(["type": /* Convert type to a NeedleValue of a string array of types, or just a single string if only 1 type in the ValueType set. */])
  }

  public static func minLength(_ value: Int) -> Self {
    Self(["minLength": .integer(value)])
  }

  // range helpers (including lengthRange) would also have additional overloads for various
  // other range types (Range, PartialRange, etc.)
  public static func range(_ range: ClosedRange<Int>) -> Self {
    Self(["minimum": .integer(range.lowerBound), "maximum": .integer(range.upperBound)])
  }

  // Have these kinds of helpers for all existing JSON schema properties...
}

public enum NeedleValue {
  // Existing cases...

  // New object case is ordered
  case object(OrderedDictionary<String, Self>)
}

// Updated Macro Usage (uses the schema merging, type inferred on a per-property basis)
// Macro type validation would be looser, but more flexible.
@NeedleGenerable(
  .title("Input"),
  .description("Sends an email."),
  // Other meta properties (oneOf, anyOf, etc.) can also be placed here.
)
struct Input: Sendable {
  @NeedleGuide(
    key: "addr", // Key override still uses the macro argument.
    .pattern("[a-z][a-z0-9]{1,10}@gmail\\.com"),
    .description("The recipient's email address."),
    .examples(["blob@blob.com", ...])
  )
  var address: String

  @NeedleGuide(.description("The subject of an email."))
  var subject: String

  @NeedleGuide(.description("The content of an email."))
  var body: String
}
```
