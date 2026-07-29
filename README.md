# Swift Edge Tools

A Swift runtime for local model tool calling, with built-in support for
[Cactus Needle](https://github.com/cactus-compute/needle).

## Package Traits

`Foundation` is enabled by default and provides conveniences such as `URL`-based model
loading and Foundation `Codable` integrations. When `Foundation` is enabled, model and
tokenizer loading use Foundation's file I/O facilities.

The `MLX`, `CoreML`, `CoreAI`, `Transformers`, and `FoundationModels` traits enable
`Foundation` automatically. Foundation can also be enabled independently:

```sh
swift build --disable-default-traits --traits Foundation
```

`XGrammar` and `Atomics` are separate traits so their dependencies are not built for a
minimal Foundation-free target. Macro support remains available in every configuration.

## Usage

```swift
import EdgeTools

let modelURL = try await NeedleMLX.download(from: .cactusNeedleMLXHuggingFace) { progress in
  print("Progress", progress)
}

// Any of (start with MLX)
let model = try NeedleMLX(from: modelURL)
let model = try NeedleONNX(from: modelURL)
let model = try NeedleCactus(from: modelURL)
let model = try NeedleCoreML(from: modelURL)
let model = try NeedleCoreAI(from: modelURL) // WWDC Rumors

let toolDefinition = EdgeToolDefinition(
  name: "get_weather",
  arguments: toolSchema
)

let response = try model.invoke(messages: [], tools: [toolDefinition])
print(response.messages, response.toolCalls)

struct GetWeather: EdgeTool {
  @EdgeToolsGenerable
  struct Input {
    let city: String
  }

  func invoke(input: sending Input) async throws -> String {
    // ...
  }
}

// Can also construct with NeedleCactus, NeedleONNX, etc.
let session = try EdgeToolsSession<NeedleMLX>(from: modelURL)

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

struct GetPopulation: EdgeTool {
  @EdgeToolsGenerable
  struct Input {
    let city: String
  }

  func invoke(input: sending Input) async throws -> Int {
    // ...
  }
}

// Dynamic
let session = try EdgeToolsSession<NeedleMLX>(from: modelURL)
let response = try await session.invoke(
  prompt: "What is the weather in San Francisco?", 
  tools: [GetWeather(), GetPopulation()]
)
print(response) // EdgeToolsSession.DynamicToolInvocations

let weatherResult = response[0].result(of: GetWeather.self) // Result<EdgeToolInvocation<GetWeather>, any Error>?
let populationResult = response[0].output(of: GetPopulation.self) // Result<EdgeToolInvocation<GetPopulation>, any Error>?

// Static

@NeedleStaticToolCollection
enum SessionTools {
  case getWeather(GetWeather)
  case getPopulation(GetPopulation)
}

// Macro Generates
// @NeedleStaticToolCollectionCase(GetWeather(/* Custom args */)) on case if one wants custom configuration
extension SessionTools {
  static let tools: [any EdgeTool] = [GetWeather(), GetPopulation()]

  enum Output: NeedleStaticToolsCollectionOutput {
    case getWeather(GetWeather.Output)
    case getPopulation(GetPopulation.Output)

    // Generates getter func body...
  }
}

let session = try EdgeToolsSession<NeedleMLX>(from: modelURL)
let response = try await session.invoke(
  prompt: "What is the weather in San Francisco?", 
  tools: SessionTools.self
)
print(response) // EdgeToolsSession.StaticToolInvocations<SessionTools.Output>
```

New JSON Schema API

```swift
let schema = EdgeToolsGenerationSchema(
  .type(.string),
  .minLength(1)
)

// Raw init (each value is a EdgeToolsValue)
let schema = EdgeToolsGenerationSchema([
  "type": "string",
  "minLength": 1
])

// Composite Keys
let schema = EdgeToolsGenerationSchema(
  .type(.string),
  .lengthRange(1...10),
  .pattern("[0-9a-zA-Z]+")
)

let schema = EdgeToolsGenerationSchema(
  .type(.number),
  .range(1.2...10.1)
)

// Arrays
let schema = EdgeToolsGenerationSchema(
  .array, // OR .type(.array)
  .properties(.type(.string), .minLength(1))
)

// Objects
let schema = EdgeToolsGenerationSchema(
  .type(.object), // OR .type(.object)
  .properties([
    "name": EdgeToolsGenerationSchema(
      .string, // OR .type(.string)
      .minLength(1)
    )
  ])
)

let schema = EdgeToolsGenerationSchema(
  .type(.object), // OR .type(.object)
  .properties([
    "name": [
      .string, // OR .type(.string)
      .minLength(1)
    ]
  ])
)

// Unions
let schema = EdgeToolsGenerationSchema(
  .type([.string, .number]),
  .minLength(1), // string constraint
  .minimum(1) // number constraint
)

// Metakeys
let schema = EdgeToolsGenerationSchema(
  .type(.number),
  .range(1.2...10.1),
  .oneOf([otherSchema, ...]),
  .allOf([otherSchema, ...])
)

public enum EdgeToolsGenerationSchema: Hashable, Sendable, Codable {
  public struct Key: RawRepresentable, ExpressibleByStringLiteral, Codable {
    // Have extensions for all common keys in JSON Schema standard ("type", "minLength", etc.)
  }

  // Ordered dictionary would guarantee ordering for Needle Prompts
  case object(OrderedDictionary<Key: EdgeToolsValue>)
  case boolean(Bool)

  public init(_ properties: OrderedDictionary<Key: EdgeToolsValue>)

  // Allows inits that you see above with strongly typed properties. Implementation would merge underlying property dicts
  public init(_ schemas: some Sequence<Self>)
  public init(_ schemas: Self...)
}

extension EdgeToolsGenerationSchema: ExpressibleByArrayLiteral {}
extension EdgeToolsGenerationSchema: ExpressibleByDictionaryLiteral {}
extension EdgeToolsGenerationSchema: ExpressibleByBooleanLiteral {}

extension EdgeToolsGenerationSchema {
  public static func type(_ type: ValueType) -> Self {
    Self(["type": /* Convert type to a EdgeToolsValue of a string array of types, or just a single string if only 1 type in the ValueType set. */])
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

public enum EdgeToolsValue {
  // Existing cases...

  // New object case is ordered
  case object(OrderedDictionary<String, Self>)
}

// Updated Macro Usage (uses the schema merging, type inferred on a per-property basis)
// Macro type validation would be looser, but more flexible.
@EdgeToolsGenerable(
  .title("Input"),
  .description("Sends an email."),
  // Other meta properties (oneOf, anyOf, etc.) can also be placed here.
)
struct Input: Sendable {
  @EdgeToolsGuide(
    key: "addr", // Key override still uses the macro argument.
    .pattern("[a-z][a-z0-9]{1,10}@gmail\\.com"),
    .description("The recipient's email address."),
    .examples(["blob@blob.com", ...])
  )
  var address: String

  @EdgeToolsGuide(.description("The subject of an email."))
  var subject: String

  @EdgeToolsGuide(.description("The content of an email."))
  var body: String
}
```

Constrained Generation

```swift
@EdgeToolsGenerable
struct MyGenerable {
  // ...    
}

@nonexhaustive
public enum XGRGenerationConstraint: Sendable {
  case unconstrained

  // We need to make XGRGrammar Sendable for this to work (should be fine to use 
  // @unchecked Sendable since it's thread-safe internally)
  case grammar(XGRGrammar)
  
  case tools(
    range: GrammarToolCallRange = .unbounded(minimum: 0), 
    grammar: (@Sendable (_ tools: XGRGrammar) -> XGRGrammar)? = nil
  )

  public static let tools = Self.tools()

  public static func schema(_ type: (some EdgeToolsGenerable).Type) -> Self {
    .grammar(.schema(type))
  }
}

let session = EdgeToolsSession(...)

// Unconstrained
let params = EdgeToolsSession.Engine.GenerateParameters(
  constraint: .unconstrained
)

// Tools
let params = EdgeToolsSession.Engine.GenerateParameters(
  constraint: .tools(range: .exact(1))
)

// Grammar (this needs a new `.schema` static init)
let params = EdgeToolsSession.Engine.GenerateParameters(
  constraint: .grammar(.schema(MyGenerable.edgeToolsGenerationSchema))
)

// Tools with potential for strongly typed response
let params = EdgeToolsSession.Engine.GenerateParameters(
  constraint: .tools { tools in union(tools, .schema(MyGenerable.self)) }
)
let params = EdgeToolsSession.Engine.GenerateParameters(
  constraint: .tools { tools in 
    union(tools, .schema(MyGenerable.edgeToolsGenerationSchema)) 
  }
)

// Typed Generation (works with any `ConvertibleFromEdgeToolsValue` type)
let value = try await session.generate(prompt: prompt).decoded(as: MyGenerable.self)

let stream = try session.stream(prompt: prompt)
let value = try await stream.decodedResponse(as: MyGenerable.self)
```
