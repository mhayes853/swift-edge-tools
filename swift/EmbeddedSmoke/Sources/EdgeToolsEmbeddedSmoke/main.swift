import EdgeTools
import OrderedCollections

struct EchoTool: EdgeTool {
  @EdgeToolsGenerable
  struct Input: Sendable {
    var message: String
  }

  let name = "echo"
  let description = "Echoes a message back to the caller."

  func invoke(input: Input) async throws -> String {
    input.message
  }
}

final class MockEngine: EdgeToolsEngine {
  final class Context: Identifiable, Sendable {}

  struct GenerateParameters: EdgeToolsEngineGenerateParameters {
    static let `default` = GenerateParameters()
    var maxTokens: Int? { nil }
  }

  final class GenerationTask: EdgeToolsEngineGenerationTask {
    private let generation: EdgeToolsEngineGeneration

    init(generation: EdgeToolsEngineGeneration) {
      self.generation = generation
    }

    var value: EdgeToolsEngineGeneration {
      get async throws { self.generation }
    }

    func stop() {}
  }

  struct Prompt: Sendable {}

  func context(_ parameters: Void) -> Context {
    Context()
  }

  func tokenize(
    prompt: Prompt,
    tools: [EdgeToolDefinition],
    context: Context
  ) async throws -> [EdgeToolsToken] {
    []
  }

  func generate(
    prompt: Prompt,
    tools: [EdgeToolDefinition],
    parameters: GenerateParameters,
    context: Context,
    channel: sending EdgeToolsGenerationChannel
  ) throws -> GenerationTask {
    let token = EdgeToolsToken(id: 0, stringValue: "calling")
    let arguments = EdgeToolsValue.object(["message": .string("hello embedded")])
    let toolCall = EdgeRawToolCall(name: "echo", arguments: arguments)
    channel.emit(token: token)
    channel.emit(part: .toolCall(toolCall))
    return GenerationTask(
      generation: EdgeToolsEngineGeneration(
        prefillMetrics: EdgeToolsPrefillMetrics(tokens: 0, duration: .zero),
        decodeMetrics: EdgeToolsDecodeMetrics(
          tokens: 1,
          duration: .zero,
          durationToFirstToken: .zero
        ),
        wasStopped: false,
        tokens: [token],
        response: token.stringValue,
        parts: [.toolCall(toolCall)]
      )
    )
  }
}

func runSmoke() async throws {
  let session = EdgeToolsSession(engine: MockEngine()) {
    EchoTool()
  }
  let generation = try await session.generate(prompt: MockEngine.Prompt(), context: nil)

  guard generation.toolCalls.count == 1 else { throw SmokeError.unexpectedToolCallCount }
  let call = generation.toolCalls[0]
  guard call.tool.name == "echo" else { throw SmokeError.unexpectedToolName }
  guard let input = call.input as? EchoTool.Input, input.message == "hello embedded" else {
    throw SmokeError.unexpectedToolInput
  }
  guard try await call.output as? String == "hello embedded" else {
    throw SmokeError.unexpectedToolOutput
  }

  let schema = EchoTool.Input.edgeToolsGenerationSchema
  let expectedSchema =
    #"{"type":"object","properties":{"message":{"type":"string"}},"required":["message"]}"#
  guard schema.orderedJSONString() == expectedSchema else { throw SmokeError.unexpectedSchema }

  let parsed = try EdgeToolsValue(json: Array(#"{"message":"round trip"}"#.utf8))
  guard try EchoTool.Input(edgeToolsValue: parsed).message == "round trip" else {
    throw SmokeError.unexpectedDecodedValue
  }
}

enum SmokeError: String, Error {
  case unexpectedToolCallCount
  case unexpectedToolName
  case unexpectedToolInput
  case unexpectedToolOutput
  case unexpectedSchema
  case unexpectedDecodedValue
}

do {
  try await runSmoke()
  print("EDGE_TOOLS_EMBEDDED_SMOKE_OK")
} catch {
  print((error as? SmokeError)?.rawValue ?? "unexpectedError")
  print("EDGE_TOOLS_EMBEDDED_SMOKE_FAILED")
}
