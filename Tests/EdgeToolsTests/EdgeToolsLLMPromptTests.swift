import CustomDump
import EdgeTools
import Testing

@Suite
struct `EdgeToolsLLMPrompt tests` {
  struct MIMETypeCase: Sendable {
    let path: String
    let mimeType: EdgeToolsMIMEType?
  }

  @Test(
    arguments: [
      MIMETypeCase(path: "image.png", mimeType: .png),
      MIMETypeCase(path: "https://example.com/image.JPG?size=large#preview", mimeType: .jpeg),
      MIMETypeCase(path: "file:///tmp/image.webp", mimeType: .webP),
      MIMETypeCase(path: "recording.MP3?download=1", mimeType: .mp3),
      MIMETypeCase(path: "recording.m4a#clip", mimeType: .m4a),
      MIMETypeCase(path: "image?name=image.png", mimeType: nil),
      MIMETypeCase(path: "image.", mimeType: nil),
      MIMETypeCase(path: "image.bmp", mimeType: .bmp)
    ]
  )
  func `MIME Type Is Inferred From Path`(_ testCase: MIMETypeCase) {
    expectNoDifference(EdgeToolsMIMEType(path: testCase.path), testCase.mimeType)
  }

  @Test
  func `Path Media Infers And Overrides MIME Type`() {
    let inferredImage = EdgeToolsLLMPrompt.Asset(path: "image.png")
    let overriddenImage = EdgeToolsLLMPrompt.Asset(path: "image.png", mimeTypeOverride: .jpeg)

    expectNoDifference(inferredImage.mimeType, .png)
    expectNoDifference(overriddenImage.mimeType, .jpeg)
    expectNoDifference(overriddenImage.path, "image.png")
  }

  @Test
  func `Byte Media Preserves MIME Type`() {
    let audio = EdgeToolsLLMPrompt.Asset(bytes: [1, 2, 3], mimeTypeOverride: .wav)
    expectNoDifference(audio.bytes, [1, 2, 3])
    expectNoDifference(audio.mimeType, .wav)
  }

  @Test
  func `Assistant Reuses Raw Tool Calls`() {
    let call = EdgeRawToolCall(name: "getWeather", arguments: ["location": "Paris"])
    let message = EdgeToolsLLMPrompt.Message.assistant(toolCalls: [call])

    expectNoDifference(message.toolCalls, [call])
  }

  @Test
  func `Tool Response Preserves Name And Structured Value`() {
    let response: EdgeToolsValue = ["temperatureCelsius": 21]
    let message = EdgeToolsLLMPrompt.Message.tool(name: "getWeather", response: response)

    expectNoDifference(message.content, nil)
    expectNoDifference(message.toolName, "getWeather")
    expectNoDifference(message.toolResponse, response)
  }
}
