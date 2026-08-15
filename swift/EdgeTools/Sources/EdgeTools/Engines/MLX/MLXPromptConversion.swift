#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && canImport(MLX)
  import OrderedCollections
  import _EdgeToolsFoundation
  import MLX
  import MLXLLM
  import MLXLMCommon
  import MLXNN
  import Observation

  #if canImport(CoreImage) && canImport(MLXVLM)
    import CoreImage
    import Foundation
    import MLXVLM
  #endif

  // MARK: - Prompt Conversion

  #if HuggingFaceTokenizers && canImport(CTokenizers)
    extension MLXLLMModelProfile where Prompt == EdgeToolsTranscript {
      public static nonisolated(nonsending) func input(
        prompt: EdgeToolsTranscript,
        tools: [EdgeToolDefinition],
        tokenizer: any EdgeToolsTokenizer,
        processor: (any UserInputProcessor)?
      ) async throws -> LMInput {
        try self.input(
          prompt: prompt,
          tools: tools,
          tokenizer: tokenizer,
          addGenerationPrompt: true
        )
      }

      public static nonisolated(nonsending) func prefillInput(
        prompt: EdgeToolsTranscript,
        tools: [EdgeToolDefinition],
        tokenizer: any EdgeToolsTokenizer,
        processor: (any UserInputProcessor)?
      ) async throws -> LMInput {
        try self.input(
          prompt: prompt,
          tools: tools,
          tokenizer: tokenizer,
          addGenerationPrompt: false
        )
      }

      private static func input(
        prompt: EdgeToolsTranscript,
        tools: [EdgeToolDefinition],
        tokenizer: any EdgeToolsTokenizer,
        addGenerationPrompt: Bool
      ) throws -> LMInput {
        guard let tokenizer = tokenizer as? any EdgeToolsChatTokenizer else {
          throw EdgeToolsError.unsupportedTokenizer
        }
        let tokenIds = try tokenizer.applyChatTemplate(
          messages: try prompt.chatTemplateMessages(),
          tools: tools.chatTemplateToolValues,
          addGenerationPrompt: addGenerationPrompt,
          additionalContext: Self.templateContext(prompt: prompt)
        )
        return LMInput(tokens: MLXArray(tokenIds))
      }
    }

    extension EdgeToolsTranscript {
      fileprivate func chatTemplateMessages() throws -> [EdgeToolsValue] {
        try self.messages.map { try $0.chatTemplateValue() }
      }
    }

    extension EdgeToolsTranscript.Message {
      public func chatTemplateValue() throws -> EdgeToolsValue {
        switch self {
        case .system(let message):
          ["role": "system", "content": .string(message.content)]
        case .user(let message):
          ["role": "user", "content": .string(message.content)]
        case .assistant(let message):
          self.assistantChatTemplateValue(parts: message.parts)
        case .tool(let message):
          [
            "role": "tool",
            "content": .string(String(decoding: try Self.encode(message.response), as: UTF8.self)),
            "name": .string(message.name)
          ]
        }
      }

      public func mlxMessage() throws -> MLXLMCommon.Message {
        try self.chatTemplateValue().mlxMessage
      }

      private func assistantChatTemplateValue(
        parts: [EdgeToolsGenerationPart]
      ) -> EdgeToolsValue {
        var message: OrderedDictionary<String, EdgeToolsValue> = ["role": "assistant"]
        let content = parts.compactMap(\.text).joined()
        let reasoning = parts.compactMap(\.reasoning).joined()
        let toolCalls = parts.compactMap(\.toolCall)
        if !content.isEmpty {
          message["content"] = .string(content)
        }
        if !reasoning.isEmpty {
          message["reasoning_content"] = .string(reasoning)
          message["thinking"] = .string(reasoning)
        }
        if !toolCalls.isEmpty {
          message["tool_calls"] = .array(toolCalls.map(\.chatTemplateValue))
        }
        return .object(message)
      }

      private static func encode(_ value: EdgeToolsValue) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(value)
      }
    }

    extension EdgeRawToolCall {
      fileprivate var chatTemplateValue: EdgeToolsValue {
        [
          "type": "function",
          "function": [
            "name": .string(self.name),
            "arguments": self.arguments
          ]
        ]
      }
    }

    extension EdgeToolDefinition {
      public var chatTemplateValue: EdgeToolsValue {
        [
          "type": "function",
          "function": [
            "name": .string(self.name),
            "description": .string(self.description),
            "parameters": self.arguments.edgeToolsValue
          ]
        ]
      }

      public var mlxToolSpec: ToolSpec {
        self.chatTemplateValue.mlxMessage
      }
    }

    extension Sequence where Element == EdgeToolDefinition {
      var chatTemplateToolValues: [EdgeToolsValue]? {
        let values = self.filter(\.includesSchemaInInstructions).map(\.chatTemplateValue)
        return values.isEmpty ? nil : values
      }

      var mlxToolSpecs: [ToolSpec]? {
        self.chatTemplateToolValues?.map(\.mlxMessage)
      }
    }

    extension EdgeToolsValue {
      fileprivate var mlxValue: any Sendable {
        switch self {
        case .array(let values): values.map(\.mlxValue)
        case .boolean(let value): value
        case .integer(let value): value
        case .null: NSNull()
        case .number(let value): value
        case .object(let object):
          Dictionary(uniqueKeysWithValues: object.map { ($0.key, $0.value.mlxValue) })
        case .string(let value): value
        }
      }

      fileprivate var mlxMessage: MLXLMCommon.Message {
        guard case .object(let object) = self else { return [:] }
        return Dictionary(uniqueKeysWithValues: object.map { ($0.key, $0.value.mlxValue) })
      }
    }

  #endif

  private enum MLXInputKind: Equatable {
    case generation
    case prefill
  }

  // MARK: - VLM Prompt Conversion

  #if canImport(CoreImage) && canImport(MLXVLM)
    private struct MLXTemporaryVideoInputs: ~Copyable {
      let videos: [UserInput.Video]
      private let temporaryURLs: [URL]

      init<Assets: Sequence>(assets: Assets) throws
      where Assets.Element == EdgeToolsTranscript.Asset {
        var videos = [UserInput.Video]()
        var temporaryURLs = [URL]()
        videos.reserveCapacity(assets.underestimatedCount)

        do {
          for asset in assets {
            switch asset.content {
            case .path(let path):
              videos.append(.url(URL(filePath: path)))
            case .bytes(let bytes):
              let url = Self.temporaryURL(for: asset.mimeType)
              try Data(bytes).write(to: url, options: .atomic)
              videos.append(.url(url))
              temporaryURLs.append(url)
            }
          }
        } catch {
          for url in temporaryURLs {
            try? FileManager.default.removeItem(at: url)
          }
          throw error
        }

        self.videos = videos
        self.temporaryURLs = temporaryURLs
      }

      deinit {
        for url in self.temporaryURLs {
          try? FileManager.default.removeItem(at: url)
        }
      }

      private static func temporaryURL(for mimeType: EdgeToolsMIMEType?) -> URL {
        let fileExtension =
          switch mimeType?.rawValue {
          case EdgeToolsMIMEType.m4v.rawValue: "m4v"
          case EdgeToolsMIMEType.quickTime.rawValue: "mov"
          default: "mp4"
          }
        let directory = FileManager.default.temporaryDirectory
        return directory.appending(path: "EdgeTools-\(UUID().uuidString).\(fileExtension)")
      }
    }

    extension EdgeToolsTranscript.Asset {
      public func mlxImage() throws -> UserInput.Image {
        switch self.content {
        case .path(let path):
          return .url(URL(filePath: path))
        case .bytes(let bytes):
          guard let image = CIImage(data: Data(bytes)) else {
            throw EdgeToolsError.invalidMedia("The image bytes could not be decoded.")
          }
          return .ciImage(image)
        }
      }
    }

    extension Sequence where Element == EdgeToolsTranscript.Asset {
      func mlxImages() throws -> [UserInput.Image] {
        try self.map { try $0.mlxImage() }
      }
    }

    extension EdgeToolsTranscript {
      func rejectAudio() throws {
        guard !self.audio.isEmpty else { return }
        throw EdgeToolsError.unsupportedMedia(
          "This MLX model integration does not support audio input."
        )
      }

      func rejectVideos() throws {
        guard !self.videos.isEmpty else { return }
        throw EdgeToolsError.unsupportedMedia(
          "This MLX model integration does not support video input."
        )
      }

      public func mlxUserInput(
        tools: [EdgeToolDefinition],
        videos: [UserInput.Video] = [],
        additionalContext: [String: EdgeToolsValue]? = nil,
        transformMessage: (Message) throws -> MLXLMCommon.Message
      ) throws -> UserInput {
        try self.rejectAudio()
        if videos.isEmpty { try self.rejectVideos() }
        return UserInput(
          messages: try self.messages.map(transformMessage),
          images: try self.images.mlxImages(),
          videos: videos,
          tools: tools.mlxToolSpecs,
          additionalContext: additionalContext?.mapValues(\.mlxValue)
        )
      }

      public func mlxVLMInput(
        tools: [EdgeToolDefinition],
        processor: any UserInputProcessor,
        additionalContext: [String: EdgeToolsValue]? = nil,
        transformMessage: (Message) throws -> MLXLMCommon.Message
      ) async throws -> LMInput {
        let videoInputs = try MLXTemporaryVideoInputs(assets: self.videos)
        return try await processor.prepare(
          input: try self.mlxUserInput(
            tools: tools,
            videos: videoInputs.videos,
            additionalContext: additionalContext
          ) { try transformMessage($0) }
        )
      }
    }
  #endif
#endif
