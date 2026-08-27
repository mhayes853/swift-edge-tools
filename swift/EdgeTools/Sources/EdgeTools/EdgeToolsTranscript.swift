import EdgeToolsCore

// MARK: - EdgeToolsTranscript

public struct EdgeToolsTranscript: Hashable, Sendable {
  public var messages: [Message]

  public init(messages: [Message] = []) {
    self.messages = messages
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

// MARK: - Prompt

extension EdgeToolsTranscript {
  /// A batch of messages to append to a transcript before generating a response.
  public struct Prompt: Hashable, Sendable {
    public var messages: [Message]

    public init(messages: [Message]) {
      self.messages = messages
    }

    public static func user(_ message: UserMessage) -> Self {
      Self(messages: [.user(message)])
    }

    public static func user(
      _ content: String,
      images: [Asset] = [],
      videos: [Asset] = [],
      audio: [Asset] = []
    ) -> Self {
      self.user(UserMessage(content: content, images: images, videos: videos, audio: audio))
    }

    public static func tools(_ responses: [ToolMessage]) -> Self {
      Self(messages: responses.map(Message.tool))
    }
  }
}

// MARK: - Message

extension EdgeToolsTranscript {
  @nonexhaustive
  public enum Message: Hashable, Sendable {
    case system(SystemMessage)
    case user(UserMessage)
    case assistant(AssistantMessage)
    case tool(ToolMessage)
  }
}

// MARK: - Message Content

extension EdgeToolsTranscript {
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

extension EdgeToolsTranscript.Message {
  public static func system(_ content: String) -> Self {
    .system(EdgeToolsTranscript.SystemMessage(content: content))
  }

  public static func user(
    _ content: String,
    images: [EdgeToolsTranscript.Asset] = [],
    videos: [EdgeToolsTranscript.Asset] = [],
    audio: [EdgeToolsTranscript.Asset] = []
  ) -> Self {
    .user(
      EdgeToolsTranscript.UserMessage(
        content: content,
        images: images,
        videos: videos,
        audio: audio
      )
    )
  }

  public static func assistant(_ parts: [EdgeToolsGenerationPart]) -> Self {
    .assistant(EdgeToolsTranscript.AssistantMessage(parts: parts))
  }

  public static func tool(name: String, response: EdgeToolsValue) -> Self {
    .tool(EdgeToolsTranscript.ToolMessage(name: name, response: response))
  }

  public init(generation: EdgeToolsEngineGeneration) {
    let usesResponse = generation.parts.isEmpty && !generation.response.isEmpty
    let parts = usesResponse ? [.text(generation.response)] : generation.parts
    self = .assistant(EdgeToolsTranscript.AssistantMessage(parts: parts))
  }
}

// MARK: - Asset

extension EdgeToolsTranscript {
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
