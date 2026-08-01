import EdgeTools
import JavaScriptKit
import Testing

@Suite(.serialized)
struct `Needle JSONNX model engine tests` {
  @Test
  func `Generates Real Tool Call`() async throws {
    let engine = try await Self.engine()
    let session = EdgeToolsSession(engine: engine, tools: [SendEmailTool()])
    let parameters = NeedleJSONNXModelEngine.GenerateParameters(
      constraint: .toolsWithGrammar(range: .exact(1)),
      maxTokens: 96
    )
    let generation = try await session.generate(
      prompt: NeedlePrompt(
        system: "",
        user: "Send an email to Henry asking him to go on an adventure."
      ),
      parameters: parameters,
      shouldInvokeTools: { _ in false }
    )

    #expect(!generation.engineGeneration.wasStopped)
    #expect(!generation.engineGeneration.tokens.isEmpty)
    #expect(generation.toolCalls.count == 1)

    let call = generation.toolCalls[0]
    #expect(call.tool.name == "sendEmail")
    let input = try #require(call.input as? SendEmailTool.Input)
    #expect(input.address.hasSuffix("@gmail.com"))
    #expect(!input.subject.isEmpty)
    #expect(input.body.lowercased().contains("adventure"))
    #expect(
      generation.response
        == generation.engineGeneration.tokens.map(\.stringValue).joined()
    )
  }

  @Test
  func `Grows Adaptive Cache Beyond Initial Capacity`() async throws {
    let engine = try await Self.engine()
    let generationTask = try engine.generate(
      prompt: NeedlePrompt(
        system: "",
        user: "Send an email to Henry asking him to go on an adventure."
      ),
      parameters: ONNXGenerateParameters(
        processor: SuppressingTokenONNXLogitsProcessor(tokenID: 1),
        constraint: .unconstrained,
        maxTokens: 130
      ),
      channel: EdgeToolsGenerationChannel()
    )
    let generation = try await generationTask.value

    let tokensAcrossCacheGrowth = generation.tokens.suffix(6).map(\.stringValue)
    #expect(!generation.wasStopped)
    #expect(generation.tokens.count == 130)
    #expect(
      tokensAcrossCacheGrowth == ["\"", "json", " []", "[", "\"", "]]"]
    )
  }

  private static func engine() async throws -> NeedleJSONNXModelEngine {
    let fixture = try #require(JSObject.global["edgeToolsNeedleFixture"].object)
    let namespace = try #require(JSObject.global["edgeToolsONNXRuntime"].object)
    let tokenizerData = try Self.bytes(named: "tokenizer.model", in: fixture)
    let tokenizer = try NeedleSPTokenizer(data: tokenizerData)
    let encoderData = try Self.file(named: "encoder.onnx.data", in: fixture)
    let decoderData = try Self.file(named: "decoder.onnx.data", in: fixture)
    return try await NeedleJSONNXModelEngine(
      onnxRuntime: namespace,
      configuration: NeedleModelConfiguration(dtype: "float32"),
      tokenizer: tokenizer,
      encoderModel: JSONNXRuntime.ModelSource.object(
        try Self.file(named: "encoder.onnx", in: fixture)
      ),
      decoderModel: JSONNXRuntime.ModelSource.object(
        try Self.file(named: "decoder.onnx", in: fixture)
      ),
      encoderConfiguration: Self.configuration(
        externalDataPath: "encoder.onnx.data",
        data: encoderData
      ),
      decoderConfiguration: Self.configuration(
        externalDataPath: "decoder.onnx.data",
        data: decoderData
      )
    )
  }

  private static func configuration(
    externalDataPath path: String,
    data: JSObject
  ) -> JSONNXRuntime.Configuration {
    let file = JSObject()
    file["path"] = .string(path)
    file["data"] = data.jsValue
    let configuration = JSONNXRuntime.Configuration()
    configuration["externalData"] = [file.jsValue].jsValue
    return configuration
  }

  private static func file(named name: String, in fixture: JSObject) throws -> JSObject {
    try #require(fixture[name].object)
  }

  private static func bytes(named name: String, in fixture: JSObject) throws -> [UInt8] {
    let object = try Self.file(named: name, in: fixture)
    try #require(object.isInstanceOf(UInt8.typedArrayClass))
    let bytes = JSUint8Array(unsafelyWrapping: object)
    return bytes.withUnsafeBytes { Array($0) }
  }
}

private struct SuppressingTokenONNXLogitsProcessor: ONNXLogitsProcessor, Sendable {
  let tokenID: EdgeToolsToken.ID

  func prompt(_ prompt: [EdgeToolsToken.ID]) {}

  func process(logits: inout MutableSpan<Float>) {
    logits[self.tokenID] = -.infinity
  }

  func didSample(tokenId: EdgeToolsToken.ID) {}
}

private struct SendEmailTool: EdgeTool {
  @EdgeToolsGenerable
  struct Input: Sendable {
    @EdgeToolsGuide(
      .pattern("[a-z][a-z0-9]{1,10}@gmail\\.com"),
      .description("The recipient's email address."),
      .examples(["blob@gmail.com"])
    )
    var address: String

    @EdgeToolsGuide(.description("The subject of an email."))
    var subject: String

    @EdgeToolsGuide(.description("The content of an email."))
    var body: String
  }

  let name = "sendEmail"
  let description = "Sends an email to a recipient with an email address."

  func invoke(input: Input) async throws -> String {
    "Sent email to \(input.address)"
  }
}
