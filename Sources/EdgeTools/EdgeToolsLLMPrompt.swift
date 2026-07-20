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
    public var toolName: String?
    public var toolResponse: EdgeToolsValue?

    public init(
      role: Role,
      content: String? = nil,
      images: [Asset]? = nil,
      audio: [Asset]? = nil,
      toolCalls: [EdgeRawToolCall]? = nil,
      toolName: String? = nil,
      toolResponse: EdgeToolsValue? = nil
    ) {
      self.role = role
      self.content = content
      self.images = images
      self.audio = audio
      self.toolCalls = toolCalls
      self.toolName = toolName
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

    public static func tool(name: String, response: EdgeToolsValue) -> Self {
      Self(role: .tool, toolName: name, toolResponse: response)
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

// MARK: - MIME Type

public struct EdgeToolsMIMEType:
  RawRepresentable,
  Hashable,
  Sendable,
  Codable,
  ExpressibleByStringLiteral
{
  public var rawValue: String

  public init(rawValue: String) {
    self.rawValue = rawValue
  }

  public init(stringLiteral value: StringLiteralType) {
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
    case "mp3": self = .mp3
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
  public static let mp3 = Self(rawValue: "audio/mpeg")
  public static let ogg = Self(rawValue: "audio/ogg")
  public static let opus = Self(rawValue: "audio/opus")
  public static let png = Self(rawValue: "image/png")
  public static let wav = Self(rawValue: "audio/wav")
  public static let webP = Self(rawValue: "image/webp")
}

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
