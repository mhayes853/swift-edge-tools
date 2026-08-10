// MARK: - EdgeToolsConversationalPrompt

public struct EdgeToolsConversationalPrompt: Hashable, Sendable {
  public var messages: [Message]
  public var reasoningEffort: EdgeToolsReasoningEffort

  public init(
    messages: [Message],
    reasoningEffort: EdgeToolsReasoningEffort = .default
  ) {
    self.messages = messages
    self.reasoningEffort = reasoningEffort
  }

  public var images: [Asset] {
    self.messages.flatMap { message -> [Asset] in
      guard case .user(let message) = message else { return [] }
      return message.images
    }
  }

  public var videos: [Asset] {
    self.messages.flatMap { message -> [Asset] in
      guard case .user(let message) = message else { return [] }
      return message.videos
    }
  }

  public var audio: [Asset] {
    self.messages.flatMap { message -> [Asset] in
      guard case .user(let message) = message else { return [] }
      return message.audio
    }
  }

}

// MARK: - Message

extension EdgeToolsConversationalPrompt {
  @nonexhaustive
  public enum Message: Hashable, Sendable {
    case system(SystemMessage)
    case user(UserMessage)
    case assistant(AssistantMessage)
    case tool(ToolMessage)
  }
}

// MARK: - Message Content

extension EdgeToolsConversationalPrompt {
  public struct SystemMessage: Hashable, Sendable {
    public var content: String

    public init(content: String) {
      self.content = content
    }

    public init(_ content: String) {
      self.init(content: content)
    }
  }

  public struct UserMessage: Hashable, Sendable {
    public var content: String
    public var images: [Asset]
    public var videos: [Asset]
    public var audio: [Asset]

    public init(
      content: String,
      images: [Asset] = [],
      videos: [Asset] = [],
      audio: [Asset] = []
    ) {
      self.content = content
      self.images = images
      self.videos = videos
      self.audio = audio
    }

    public init(
      _ content: String,
      images: [Asset] = [],
      videos: [Asset] = [],
      audio: [Asset] = []
    ) {
      self.init(content: content, images: images, videos: videos, audio: audio)
    }
  }

  public struct AssistantMessage: Hashable, Sendable {
    public var parts: [EdgeToolsGenerationPart]

    public init(parts: [EdgeToolsGenerationPart]) {
      self.parts = parts
    }

    public init(_ parts: [EdgeToolsGenerationPart]) {
      self.init(parts: parts)
    }
  }

  public struct ToolMessage: Hashable, Sendable {
    public var name: String
    public var response: EdgeToolsValue

    public init(name: String, response: EdgeToolsValue) {
      self.name = name
      self.response = response
    }
  }
}

extension EdgeToolsConversationalPrompt.Message {
  public static func system(_ content: String) -> Self {
    .system(EdgeToolsConversationalPrompt.SystemMessage(content: content))
  }

  public static func user(
    _ content: String,
    images: [EdgeToolsConversationalPrompt.Asset] = [],
    videos: [EdgeToolsConversationalPrompt.Asset] = [],
    audio: [EdgeToolsConversationalPrompt.Asset] = []
  ) -> Self {
    .user(
      EdgeToolsConversationalPrompt.UserMessage(
        content: content,
        images: images,
        videos: videos,
        audio: audio
      )
    )
  }

  public static func assistant(_ parts: [EdgeToolsGenerationPart]) -> Self {
    .assistant(EdgeToolsConversationalPrompt.AssistantMessage(parts: parts))
  }

  public static func tool(name: String, response: EdgeToolsValue) -> Self {
    .tool(EdgeToolsConversationalPrompt.ToolMessage(name: name, response: response))
  }

  public init(generation: EdgeToolsEngineGeneration) {
    let isEmpty = generation.parts.isEmpty && !generation.response.isEmpty
    let parts = isEmpty ? [.text(generation.response)]  : generation.parts
    self = .assistant(EdgeToolsConversationalPrompt.AssistantMessage(parts: parts))
  }
}

// MARK: - Asset

extension EdgeToolsConversationalPrompt {
  public struct Asset: Hashable, Sendable {
    @nonexhaustive
    public enum Content: Hashable, Sendable {
      case path(String)
      case bytes([UInt8])
    }

    public var content: Content
    public var mimeType: EdgeToolsMIMEType?

    public init(content: Content, mimeType: EdgeToolsMIMEType?) {
      self.content = content
      self.mimeType = mimeType
    }

    public init(path: String, mimeTypeOverride: EdgeToolsMIMEType? = nil) {
      self.init(
        content: .path(path),
        mimeType: mimeTypeOverride ?? EdgeToolsMIMEType(path: path)
      )
    }

    public init(bytes: [UInt8], mimeTypeOverride: EdgeToolsMIMEType? = nil) {
      self.init(content: .bytes(bytes), mimeType: mimeTypeOverride)
    }
  }
}

// MARK: - MIME Type

public struct EdgeToolsMIMEType:
  RawRepresentable,
  Hashable,
  Sendable,
  ExpressibleByStringLiteral
{
  public var rawValue: String

  public init(rawValue: String) {
    self.rawValue = rawValue
  }

  public init(stringLiteral value: String) {
    self.init(rawValue: value)
  }

  public init?(path: String) {
    guard let pathExtension = inferredPathExtension(from: path) else { return nil }
    switch pathExtension {
    case "aac": self = .aac
    case "bmp": self = .bmp
    case "flac": self = .flac
    case "gif": self = .gif
    case "heic": self = .heic
    case "heif": self = .heif
    case "jpg", "jpeg": self = .jpeg
    case "m4a": self = .m4a
    case "m4v": self = .m4v
    case "mp3": self = .mp3
    case "mp4": self = .mp4
    case "mov": self = .quickTime
    case "ogg": self = .ogg
    case "opus": self = .opus
    case "png": self = .png
    case "wav": self = .wav
    case "webp": self = .webP
    default: return nil
    }
  }

  public static let aac = Self(rawValue: "audio/aac")
  public static let bmp = Self(rawValue: "image/bmp")
  public static let flac = Self(rawValue: "audio/flac")
  public static let gif = Self(rawValue: "image/gif")
  public static let heic = Self(rawValue: "image/heic")
  public static let heif = Self(rawValue: "image/heif")
  public static let jpeg = Self(rawValue: "image/jpeg")
  public static let m4a = Self(rawValue: "audio/mp4")
  public static let m4v = Self(rawValue: "video/x-m4v")
  public static let mp3 = Self(rawValue: "audio/mpeg")
  public static let mp4 = Self(rawValue: "video/mp4")
  public static let ogg = Self(rawValue: "audio/ogg")
  public static let opus = Self(rawValue: "audio/opus")
  public static let png = Self(rawValue: "image/png")
  public static let quickTime = Self(rawValue: "video/quicktime")
  public static let wav = Self(rawValue: "audio/wav")
  public static let webP = Self(rawValue: "image/webp")
}

extension EdgeToolsMIMEType: EdgeToolsCodable {}

private func inferredPathExtension(from path: String) -> String? {
  let path =
    path
    .split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false)[0]
    .split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false)[0]
  guard
    let component = path.split(separator: "/").last,
    let separator = component.lastIndex(of: ".")
  else { return nil }

  let pathExtension = component[component.index(after: separator)...]
  guard !pathExtension.isEmpty else { return nil }
  return pathExtension.lowercased()
}
