import EdgeTools
import JavaScriptKit
import Testing

@Suite(.serialized)
struct `Needle JSONNX engine tests` {
  @Test
  func `Generates Real Tool Call`() async throws {
    let fixture = try #require(JSObject.global["edgeToolsNeedleFixture"].object)
    let namespace = try #require(JSObject.global["edgeToolsONNXRuntime"].object)
    let tokenizerData = try Self.bytes(named: "tokenizer.model", in: fixture)
    let tokenizer = try NeedleSPTokenizer(data: tokenizerData)
    let encoderData = try Self.file(named: "encoder.onnx.data", in: fixture)
    let decoderData = try Self.file(named: "decoder.onnx.data", in: fixture)
    let engine = try await NeedleJSONNXEngine(
      onnxRuntime: namespace,
      configuration: NeedleModelConfiguration(dtype: "float32"),
      tokenizer: tokenizer,
      encoderModel: JSONNXRuntime.ModelSource.javaScriptFile(
        try Self.file(named: "encoder.onnx", in: fixture)
      ),
      decoderModel: JSONNXRuntime.ModelSource.javaScriptFile(
        try Self.file(named: "decoder.onnx", in: fixture)
      ),
      encoderConfiguration: JSONNXRuntime.Configuration(
        externalData: [
          JSONNXRuntime.ExternalDataFile(
            path: "encoder.onnx.data",
            data: JSONNXRuntime.ModelSource.javaScriptFile(encoderData)
          )
        ]
      ),
      decoderConfiguration: JSONNXRuntime.Configuration(
        externalData: [
          JSONNXRuntime.ExternalDataFile(
            path: "decoder.onnx.data",
            data: JSONNXRuntime.ModelSource.javaScriptFile(decoderData)
          )
        ]
      )
    )
    let session = EdgeToolsSession(engine: engine, tools: [SendEmailTool()])
    let parameters = NeedleJSONNXEngine.GenerateParameters(
      constraint: .tools(range: .exact(1)),
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
    #expect(input.body.localizedCaseInsensitiveContains("adventure"))
    #expect(
      generation.response
        == generation.engineGeneration.tokens.map(\.stringValue).joined()
    )
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
