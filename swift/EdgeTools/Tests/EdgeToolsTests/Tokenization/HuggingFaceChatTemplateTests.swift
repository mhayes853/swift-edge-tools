#if HuggingFaceTokenizers
  import CustomDump
  import EdgeTools
  import Foundation
  import Testing

  private struct ChatTemplateFixture: Encodable {
    let name: String
    let template: String
  }

  private struct TokenizerConfigurationFixture: Encodable {
    let bosToken: String
    let eosToken: String
    let chatTemplate: [ChatTemplateFixture]

    enum CodingKeys: String, CodingKey {
      case bosToken = "bos_token"
      case eosToken = "eos_token"
      case chatTemplate = "chat_template"
    }
  }

  @Suite
  struct `HuggingFaceChatTemplate tests` {
    @Test
    func `Renders Default And Tool Templates With Generation Prompts`() throws {
      let tokenizer = try self.makeTokenizer()
      let messages: [EdgeToolsValue] = [["role": "user", "content": "offline"]]
      let tools: [EdgeToolsValue] = [["type": "function"]]

      expectNoDifference(
        try tokenizer.renderChatTemplate(
          messages: messages,
          tools: nil,
          addGenerationPrompt: false
        ),
        "offline"
      )
      expectNoDifference(
        try tokenizer.renderChatTemplate(
          messages: messages,
          tools: tools,
          addGenerationPrompt: true
        ),
        "tools=1;offline<eos>"
      )
      expectNoDifference(
        try tokenizer.applyChatTemplate(
          messages: messages,
          tools: nil,
          addGenerationPrompt: false
        ),
        [4]
      )
      expectNoDifference(
        try tokenizer.applyChatTemplate(
          messages: messages,
          tools: nil,
          addGenerationPrompt: true
        ),
        [4, 2]
      )
    }

    @Test
    func `Additional Context Overrides Configured Special Tokens`() throws {
      let tokenizer = try self.makeTokenizer()
      expectNoDifference(
        try tokenizer.renderChatTemplate(
          messages: [],
          tools: nil,
          addGenerationPrompt: false,
          additionalContext: ["eos_token": "<override>"]
        ),
        ""
      )
      expectNoDifference(
        try tokenizer.renderChatTemplate(
          messages: [],
          tools: nil,
          addGenerationPrompt: true,
          additionalContext: ["eos_token": "<override>"]
        ),
        "<override>"
      )
    }

    @Test
    func `Duplicate Template Names Resolve To The First Match`() throws {
      let tokenizer = try HuggingFaceTokenizer(
        tokenizerJSON: try self.tokenizerJSON(),
        configuration: try JSONEncoder().encode(
          TokenizerConfigurationFixture(
            bosToken: "<bos>",
            eosToken: "<eos>",
            chatTemplate: [
              ChatTemplateFixture(name: "default", template: "first"),
              ChatTemplateFixture(name: "default", template: "second"),
            ]
          )
        )
      )
      expectNoDifference(
        try tokenizer.renderChatTemplate(messages: [], tools: nil, addGenerationPrompt: false),
        "first"
      )
    }

    @Test
    func `Missing Chat Template Throws`() throws {
      let tokenizer = try HuggingFaceTokenizer(tokenizerJSON: try self.tokenizerJSON())
      #expect(throws: EdgeToolsError.self) {
        try tokenizer.renderChatTemplate(messages: [], tools: nil, addGenerationPrompt: false)
      }
    }

    private func makeTokenizer() throws -> HuggingFaceTokenizer {
      try HuggingFaceTokenizer(
        tokenizerJSON: try self.tokenizerJSON(),
        configuration: try self.chatTemplateConfiguration()
      )
    }

    private func chatTemplateConfiguration() throws -> Data {
      try JSONEncoder().encode(
        TokenizerConfigurationFixture(
          bosToken: "<bos>",
          eosToken: "<eos>",
          chatTemplate: [
            ChatTemplateFixture(
              name: "default",
              template: defaultTemplate
            ),
            ChatTemplateFixture(
              name: "tool_use",
              template: toolUseTemplate
            ),
          ]
        )
      )
    }

    private func tokenizerJSON() throws -> Data {
      try Data(
        contentsOf: Bundle.module.url(forResource: "test_tokenizer", withExtension: "json")!
      )
    }
  }

  private let defaultTemplate =
    "{% for message in messages %}"
    + "{{ message['content'] }}"
    + "{% endfor %}"
    + "{% if add_generation_prompt %}{{ eos_token }}{% endif %}"

  private let toolUseTemplate =
    "tools={{ tools | length }};"
    + "{% for message in messages %}"
    + "{{ message['content'] }}"
    + "{% endfor %}"
    + "{% if add_generation_prompt %}<eos>{% endif %}"
#endif
