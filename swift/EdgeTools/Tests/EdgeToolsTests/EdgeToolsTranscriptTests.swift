import CustomDump
import EdgeTools
import Testing

@Suite
struct `EdgeToolsTranscript tests` {
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
}
