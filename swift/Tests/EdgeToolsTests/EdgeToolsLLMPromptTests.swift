import CustomDump
import EdgeTools
import Testing

@Suite
struct `EdgeToolsConversationalPrompt tests` {
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
      MIMETypeCase(path: "clip.MP4?download=1", mimeType: .mp4),
      MIMETypeCase(path: "clip.mov#preview", mimeType: .quickTime),
      MIMETypeCase(path: "clip.m4v", mimeType: .m4v),
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
    let inferredImage = EdgeToolsConversationalPrompt.Asset(path: "image.png")
    let overriddenImage = EdgeToolsConversationalPrompt.Asset(
      path: "image.png",
      mimeTypeOverride: .jpeg
    )

    expectNoDifference(inferredImage.mimeType, .png)
    expectNoDifference(overriddenImage.mimeType, .jpeg)
  }

  @Test
  func `Message Cases Store Dedicated Content Structs`() {
    let system = EdgeToolsConversationalPrompt.SystemMessage(content: "Instructions")
    let user = EdgeToolsConversationalPrompt.UserMessage(content: "Hello")
    let assistant = EdgeToolsConversationalPrompt.AssistantMessage(parts: [.text("Hi")])
    let tool = EdgeToolsConversationalPrompt.ToolMessage(
      name: "weather",
      response: ["temperature": .integer(72)]
    )

    let messages: [EdgeToolsConversationalPrompt.Message] = [
      .system(system),
      .user(user),
      .assistant(assistant),
      .tool(tool)
    ]

    expectNoDifference(messages.count, 4)
    expectNoDifference(messages.compactMap { message in
      guard case .user(let message) = message else { return nil }
      return message.content
    }, ["Hello"])
  }
}
