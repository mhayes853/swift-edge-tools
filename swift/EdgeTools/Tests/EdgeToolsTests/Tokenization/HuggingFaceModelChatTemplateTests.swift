#if HuggingFaceTokenizers
  import EdgeTools
  import Foundation
  import SnapshotTesting
  import Testing

  @Suite
  struct `HuggingFaceModelChatTemplate tests` {
    @Test
    func `Renders Model Profile Templates For Messages And Tools`() throws {
      let outputs = try Self.profileNames.map(Self.renderProfile)
      assertSnapshot(of: outputs.joined(separator: "\n\n"), as: .lines)
    }

    private static let profileNames = [
      "FunctionGemma",
      "Gemma4",
      "Granite",
      "LFM2P5",
      "LFM2P5Thinking",
      "LFM2P5VL",
      "MiniCPM5",
      "Qwen3",
      "Qwen3P5",
    ]

    private static func renderProfile(_ name: String) throws -> String {
      let tokenizer = try HuggingFaceTokenizer(
        tokenizerJSON: try tokenizerJSON(),
        configuration: try resourceData(name: name, file: "tokenizer_config.json"),
        chatTemplate: try chatTemplate(name: name)
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
        addGenerationPrompt: false
      )
      let promptWithGeneration = try tokenizer.renderChatTemplate(
        messages: messages,
        tools: tools,
        addGenerationPrompt: true
      )
      expectNoDifference(promptWithoutGeneration.isEmpty, false)
      expectNoDifference(promptWithGeneration.isEmpty, false)
      return "# \(name)\n\n## No Generation\n\(promptWithoutGeneration)\n\n## Generation\n\(promptWithGeneration)"
    }

    private static func chatTemplate(name: String) throws -> String? {
      guard let url = Bundle.module.url(
        forResource: "\(name)-chat_template",
        withExtension: "jinja"
      ) else {
        return nil
      }
      return try String(contentsOf: url, encoding: .utf8)
        .replacing(/strftime_now\([^)]*\)/, with: "'2020-01-02'")
    }

    private static func resourceData(name: String, file: String) throws -> Data {
      try Data(
        contentsOf: Bundle.module.url(
          forResource: "\(name)-\(file.dropLast(5))",
          withExtension: "json"
        )!
      )
    }

    private static func tokenizerJSON() throws -> Data {
      try Data(contentsOf: Bundle.module.url(forResource: "test_tokenizer", withExtension: "json")!)
    }
  }
#endif
