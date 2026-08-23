#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && canImport(MLX)
  import EdgeToolsCore
  import EdgeToolsTokenizers
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
        let tokens = try tokenizer.applyChatTemplate(
          messages: try prompt.chatTemplateMessages(),
          tools: tools.chatTemplateToolValues,
          addGenerationPrompt: addGenerationPrompt,
          additionalContext: Self.templateContext(prompt: prompt)
        )
        return LMInput(tokens: MLXArray(tokens.map(\.id)))
      }
    }

    extension EdgeToolsTranscript.Message {
      func mlxMessage() throws -> MLXLMCommon.Message {
        try self.chatTemplateValue().mlxMessage
      }
    }

    extension EdgeToolDefinition {
      var mlxToolSpec: ToolSpec {
        self.chatTemplateValue.mlxMessage
      }
    }

    extension Sequence where Element == EdgeToolDefinition {
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
      func mlxImage() throws -> UserInput.Image {
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

      func mlxUserInput(
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

      func mlxVLMInput(
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
