// MARK: - EdgeToolsLLMPrompt

public struct EdgeToolsLLMPrompt: Hashable, Sendable {
  public var messages: [Message]

  public init(messages: [Message]) {
    self.messages = messages
  }
}

// MARK: - Message

extension EdgeToolsLLMPrompt {
  public struct Message: Hashable, Sendable {
    public struct Role: RawRepresentable, Hashable, Sendable, Codable, ExpressibleByStringLiteral {
      public var rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public init(stringLiteral value: StringLiteralType) {
        self.init(rawValue: value)
      }

      public static let system = Self(rawValue: "system")
      public static let user = Self(rawValue: "user")
      public static let assistant = Self(rawValue: "assistant")
      public static let tool = Self(rawValue: "tool")
    }

    public var role: Role
    public var content: String?
    public var images: [Asset]?
    public var audio: [Asset]?
    public var toolCalls: [EdgeRawToolCall]?
    public var toolResponse: EdgeToolsValue?

    public init(
      role: Role,
      content: String? = nil,
      images: [Asset]? = nil,
      audio: [Asset]? = nil,
      toolCalls: [EdgeRawToolCall]? = nil,
      toolResponse: EdgeToolsValue? = nil
    ) {
      self.role = role
      self.content = content
      self.images = images
      self.audio = audio
      self.toolCalls = toolCalls
      self.toolResponse = toolResponse
    }

    public static func system(_ content: String) -> Self {
      Self(role: .system, content: content)
    }

    public static func user(
      _ content: String,
      images: [Asset] = [],
      audio: [Asset] = []
    ) -> Self {
      Self(
        role: .user,
        content: content,
        images: images.nilIfEmpty,
        audio: audio.nilIfEmpty
      )
    }

    public static func assistant(
      _ content: String? = nil,
      toolCalls: [EdgeRawToolCall] = []
    ) -> Self {
      Self(
        role: .assistant,
        content: content,
        toolCalls: toolCalls.nilIfEmpty
      )
    }

    public static func tool(_ response: EdgeToolsValue) -> Self {
      Self(role: .tool, toolResponse: response)
    }
  }
}

// MARK: - Asset

extension EdgeToolsLLMPrompt {
  public struct Asset: Hashable, Sendable {
    public var path: String?
    public var bytes: [UInt8]?
    public var mimeType: EdgeToolsMIMEType?

    public init(path: String, mimeTypeOverride: EdgeToolsMIMEType? = nil) {
      self.path = path
      self.bytes = nil
      self.mimeType = mimeTypeOverride ?? EdgeToolsMIMEType(path: path)
    }

    public init(bytes: [UInt8], mimeTypeOverride: EdgeToolsMIMEType? = nil) {
      self.path = nil
      self.bytes = bytes
      self.mimeType = mimeTypeOverride
    }
  }
}

// MARK: - Helpers

extension Collection {
  fileprivate var nilIfEmpty: Self? {
    self.isEmpty ? nil : self
  }
}
