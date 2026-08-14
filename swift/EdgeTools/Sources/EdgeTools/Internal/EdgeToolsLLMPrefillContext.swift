#if Foundation
  import _EdgeToolsFoundation

  // MARK: - EdgeToolsLLMPrefillContext

  struct EdgeToolsLLMPrefillContext: Hashable {
    private enum Message: Hashable {
      case system(String)
      case user(String, images: [Asset], videos: [Asset], audio: [Asset])
      case assistant([EdgeToolsGenerationPart])
      case tool(name: String, response: EdgeToolsValue)
    }

    private struct Media: Hashable {
      let images: [Asset]
      let videos: [Asset]
      let audio: [Asset]
    }

    private enum Asset: Hashable {
      case bytes([UInt8], mimeType: EdgeToolsMIMEType?)
      case file(path: String, modificationDate: Date)

      init(_ asset: EdgeToolsTranscript.Asset) {
        switch asset.content {
        case .bytes(let bytes):
          self = .bytes(bytes, mimeType: asset.mimeType)
        case .path(let path):
          let url = URL(filePath: path).standardizedFileURL
          let attributes = try? FileManager.default.attributesOfItem(atPath: url.path())
          let modificationDate =
            if (attributes?[.type] as? FileAttributeType) == .typeRegular {
              attributes?[.modificationDate] as? Date ?? Date(timeIntervalSince1970: 0)
            } else {
              Date(timeIntervalSince1970: 0)
            }
          self = .file(path: url.path(), modificationDate: modificationDate)
        }
      }
    }

    private let messages: [Message]
    private let media: [Media]
    private let tools: [EdgeToolDefinition]

    init(prompt: EdgeToolsTranscript, tools: [EdgeToolDefinition]) {
      let messages: [Message] = prompt.messages.map { message in
        switch message {
        case .system(let message):
          .system(message.content)
        case .user(let message):
          .user(
            message.content,
            images: message.images.map(Asset.init),
            videos: message.videos.map(Asset.init),
            audio: message.audio.map(Asset.init)
          )
        case .assistant(let message):
          .assistant(message.parts)
        case .tool(let message):
          .tool(name: message.name, response: message.response)
        }
      }
      self.messages = messages
      self.media = messages.map { message in
        guard case .user(_, let images, let videos, let audio) = message else {
          return Media(images: [], videos: [], audio: [])
        }
        return Media(images: images, videos: videos, audio: audio)
      }
      self.tools = tools
    }

    func hasMediaPrefix(in other: Self) -> Bool {
      self.media.count <= other.media.count
        && zip(self.media, other.media).allSatisfy(==)
    }
  }
#endif
