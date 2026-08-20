import EdgeTools
import JavaScriptKit
import Testing

@Suite(.serialized)
struct `Needle2JSEngine tests` {
  @Test
  func `Loads TypeScript System Defaults`() async {
    let system = await Needle2System.platformDefaults(
      battery: { 62.5 },
      location: { "test location" }
    )

    #expect(system[.date] != nil)
    #expect(system[.locale] != nil)
    #expect(system[.device] == "desktop")
    #expect(system[.battery] == "62.5%")
    #expect(system[.network] == nil)
    #expect(system[.location] == "test location")

    let suppressedDevice = await Needle2System.platformDefaults(device: { nil })
    #expect(suppressedDevice[.device] == nil)
  }

  @Test
  func `Generates Real Tool Call Through Worker Runtime`() async throws {
    let createRuntime = try #require(JSObject.global["edgeToolsNeedle2Runtime"].object)
    let engine = try await Needle2JSEngine(createRuntime: createRuntime)
    let context = engine.context { Needle2SendEmailTool() }
    let session = EdgeToolsSession(engine: engine)

    let generation = try await session.generate(
      prompt: "Send an email to blob@gmail.com asking them to go hiking.",
      context: context,
      shouldInvokeTools: { _ in false }
    )

    #expect(!generation.engineGeneration.wasStopped)
    #expect(!context.isResponding)
    #expect(generation.engineGeneration.tokens.isEmpty)
    #expect(generation.engineGeneration.toolCalls.count == 1)
    #expect(generation.engineGeneration.toolCalls.first?.name == "send_email")
    #expect(generation.engineGeneration.decodeMetrics.tokens > 0)
    #expect(generation.engineGeneration.metadata.needle2ResponseType == "call")
    #expect(generation.engineGeneration.metadata.needle2PrefillTokensPerSecond != nil)
    #expect(generation.engineGeneration.metadata.needle2DecodeTokensPerSecond != nil)
    #expect(generation.engineGeneration.metadata.needle2PeakRAMMegabytes == nil)

    let call = try #require(generation.toolCalls.first)
    let input = try #require(call.input as? Needle2SendEmailTool.Input)
    #expect(input.address == "blob@gmail.com")
    #expect(input.body.lowercased().contains("hiking"))
  }
}

private struct Needle2SendEmailTool: EdgeTool {
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
