#if HuggingFaceTokenizers
  import EdgeTools
  import Foundation
  import SnapshotTesting
  import Testing

  @Suite
  struct `HuggingFaceModelChatTemplate tests` {
    @Test
    func `Renders Model Profile Templates For Messages And Tools`() async throws {
      var outputs: [String] = []
      for profile in Self.models {
        outputs.append(try await Self.renderProfile(profile))
      }
      assertSnapshot(of: outputs.joined(separator: "\n\n"), as: .lines)
    }

    private static let pinnedNow: [String: EdgeToolsValue] = ["edge_tools_now": 1_577_923_200]

    private static let models: [(name: String, id: ModelID)] = [
      ("FunctionGemma", .functionGemma),
      ("Gemma4", .gemma4E2B),
      ("Granite", .graniteMoeHybrid),
      ("LFM2P5", .lfm2P5),
      ("LFM2P5Thinking", .lfm2P5Thinking),
      ("LFM2P5VL", .lfm2P5VL),
      ("MiniCPM5", .miniCPM5),
      ("Qwen3", .qwen3),
      ("Qwen3P5", .qwen3P5),
    ]

    private static func renderProfile(_ profile: (name: String, id: ModelID)) async throws -> String {
      let directory = try await downloadModel(id: profile.id)
      let tokenizer = try HuggingFaceTokenizer(
        tokenizerJSON: try Data(contentsOf: directory.appending(path: "tokenizer.json")),
        configuration: try Data(contentsOf: directory.appending(path: "tokenizer_config.json")),
        chatTemplate: try Self.chatTemplate(in: directory)
      )
      let messages: [EdgeToolsValue] = [
        ["role": "system", "content": "You are concise."],
        ["role": "user", "content": "What is the weather?"],
        ["role": "assistant", "content": "I will check."],
      ]
      let tools: [EdgeToolsValue] = [[
        "type": "function",
        "function": [
          "name": "weather",
          "description": "Gets the weather.",
          "parameters": [
            "type": "object",
            "properties": ["location": ["type": "string"]],
          ],
        ],
      ]]
      let promptWithoutGeneration = try tokenizer.renderChatTemplate(
        messages: messages,
        tools: tools,
        addGenerationPrompt: false,
        additionalContext: pinnedNow
      )
      let promptWithGeneration = try tokenizer.renderChatTemplate(
        messages: messages,
        tools: tools,
        addGenerationPrompt: true,
        additionalContext: pinnedNow
      )
      expectNoDifference(promptWithoutGeneration.isEmpty, false)
      expectNoDifference(promptWithGeneration.isEmpty, false)
      return "# \(profile.name)\n\n## No Generation\n\(promptWithoutGeneration)\n\n## Generation\n\(promptWithGeneration)"
    }

    private static func chatTemplate(in directory: URL) throws -> String? {
      let url = directory.appending(path: "chat_template.jinja")
      guard FileManager.default.fileExists(atPath: url.path()) else {
        return nil
      }
      return try String(contentsOf: url, encoding: .utf8)
    }
  }
#endif
