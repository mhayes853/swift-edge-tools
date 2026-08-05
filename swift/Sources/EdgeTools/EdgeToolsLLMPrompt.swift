// MARK: - EdgeToolsLLMPrompt

public struct EdgeToolsLLMPrompt: Hashable, Sendable {
  public var messages: [Message]

  public init(messages: [Message]) {
    self.messages = messages
  }
}

// MARK: - Message

extension EdgeToolsLLMPrompt {
  @nonexhaustive
  public enum Message: Hashable, Sendable {
    case system(String)
    case user(String, images: [Asset] = [], audio: [Asset] = [])
    case assistant(String? = nil, toolCalls: [EdgeRawToolCall] = [])
    case tool(name: String, response: EdgeToolsValue)
  }
}

// MARK: - Asset

extension EdgeToolsLLMPrompt {
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

#if !$Embedded
  extension EdgeToolsMIMEType: Codable {}
#endif

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
