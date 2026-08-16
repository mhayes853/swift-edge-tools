#if HuggingFaceTokenizers && FoundationEssentials && canImport(CTokenizers)
  import OrderedCollections
  import _EdgeToolsFoundation

  // MARK: - Hugging Face Tokenizer Configuration

  struct HuggingFaceTokenizerConfiguration: Sendable {
    let bosToken: String?
    let eosToken: String?
    let unknownToken: String?
    let sepToken: String?
    let padToken: String?
    let clsToken: String?
    let maskToken: String?
    let additionalSpecialTokens: [String]
    private let chatTemplates: [NamedChatTemplate]

    init(data: Data?, chatTemplateOverride: String? = nil) throws {
      let source = try data.map { try JSONDecoder().decode(Source.self, from: $0) } ?? Source()
      self.bosToken = source.bosToken?.content
      self.eosToken = source.eosToken?.content
      self.unknownToken = source.unknownToken?.content
      self.sepToken = source.sepToken?.content
      self.padToken = source.padToken?.content
      self.clsToken = source.clsToken?.content
      self.maskToken = source.maskToken?.content
      self.additionalSpecialTokens = source.additionalSpecialTokens.compactMap(\.content)
      self.chatTemplates = (chatTemplateOverride.map { [Source.ChatTemplate.literal($0)] }
        ?? source.chatTemplate)
        .map { NamedChatTemplate(source: $0) }
    }

    func renderChatTemplate(
      messages: [EdgeToolsValue],
      tools: [EdgeToolsValue]?,
      addGenerationPrompt: Bool,
      additionalContext: [String: EdgeToolsValue]?
    ) throws -> String {
      let source = try self.selectedChatTemplate(tools: tools)
      var context = OrderedDictionary<String, EdgeToolsValue>()
      self.addSpecialTokens(to: &context)
      context["messages"] = .array(messages)
      context["add_generation_prompt"] = .boolean(addGenerationPrompt)
      if let tools {
        context["tools"] = .array(tools)
      }
      context.merge(additionalContext ?? [:]) { _, override in override }
      return try nativeRenderTemplate(source, context: .object(context))
    }

    private func selectedChatTemplate(tools: [EdgeToolsValue]?) throws -> String {
      let toolTemplate =
        tools?.isEmpty == false
        ? self.chatTemplates.first { $0.name == toolUseChatTemplateName }
        : nil
      guard let template = toolTemplate
        ?? self.chatTemplates.first(where: { $0.name == defaultChatTemplateName })
      else {
        throw EdgeToolsError(
          code: .unsupportedTokenizer,
          message: self.chatTemplates.isEmpty
            ? "The Hugging Face tokenizer does not define a chat template."
            : "The Hugging Face tokenizer has no default chat template."
        )
      }
      return template.source
    }

    private func addSpecialTokens(to context: inout OrderedDictionary<String, EdgeToolsValue>) {
      if let bosToken { context["bos_token"] = .string(bosToken) }
      if let eosToken { context["eos_token"] = .string(eosToken) }
      if let unknownToken { context["unk_token"] = .string(unknownToken) }
      if let sepToken { context["sep_token"] = .string(sepToken) }
      if let padToken { context["pad_token"] = .string(padToken) }
      if let clsToken { context["cls_token"] = .string(clsToken) }
      if let maskToken { context["mask_token"] = .string(maskToken) }
      if !additionalSpecialTokens.isEmpty {
        context["additional_special_tokens"] = .array(
          additionalSpecialTokens.map(EdgeToolsValue.string)
        )
      }
    }
  }

  // MARK: - Chat Templates

  private let defaultChatTemplateName = "default"
  private let toolUseChatTemplateName = "tool_use"

  private struct NamedChatTemplate: Sendable {
    let name: String
    let source: String

    init(source: Source.ChatTemplate) {
      switch source {
      case .literal(let template):
        self.name = defaultChatTemplateName
        self.source = template
      case .named(let name, let template):
        self.name = name
        self.source = template
      }
    }
  }

  // MARK: - Configuration Source

  private struct Source: Decodable {
    struct Token: Decodable {
      let content: String

      init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
          self.content = value
        } else {
          self.content = try container.decode(Content.self).content
        }
      }
    }

    struct Content: Decodable {
      let content: String
    }

    enum ChatTemplate: Decodable {
      case literal(String)
      case named(name: String, template: String)

      private enum CodingKeys: String, CodingKey {
        case name
        case template
      }

      init(from decoder: any Decoder) throws {
        if let value = try? decoder.singleValueContainer().decode(String.self) {
          self = .literal(value)
          return
        }
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self = .named(
          name: try container.decode(String.self, forKey: .name),
          template: try container.decode(String.self, forKey: .template)
        )
      }
    }

    let bosToken: Token?
    let eosToken: Token?
    let unknownToken: Token?
    let sepToken: Token?
    let padToken: Token?
    let clsToken: Token?
    let maskToken: Token?
    let additionalSpecialTokens: [Token]
    let chatTemplate: [ChatTemplate]

    private enum CodingKeys: String, CodingKey {
      case bosToken = "bos_token"
      case eosToken = "eos_token"
      case unknownToken = "unk_token"
      case sepToken = "sep_token"
      case padToken = "pad_token"
      case clsToken = "cls_token"
      case maskToken = "mask_token"
      case additionalSpecialTokens = "additional_special_tokens"
      case chatTemplate = "chat_template"
    }

    init() {
      self.bosToken = nil
      self.eosToken = nil
      self.unknownToken = nil
      self.sepToken = nil
      self.padToken = nil
      self.clsToken = nil
      self.maskToken = nil
      self.additionalSpecialTokens = []
      self.chatTemplate = []
    }

    init(from decoder: any Decoder) throws {
      let container = try decoder.container(keyedBy: CodingKeys.self)
      self.bosToken = try container.decodeIfPresent(Token.self, forKey: .bosToken)
      self.eosToken = try container.decodeIfPresent(Token.self, forKey: .eosToken)
      self.unknownToken = try container.decodeIfPresent(Token.self, forKey: .unknownToken)
      self.sepToken = try container.decodeIfPresent(Token.self, forKey: .sepToken)
      self.padToken = try container.decodeIfPresent(Token.self, forKey: .padToken)
      self.clsToken = try container.decodeIfPresent(Token.self, forKey: .clsToken)
      self.maskToken = try container.decodeIfPresent(Token.self, forKey: .maskToken)
      self.additionalSpecialTokens =
        try container.decodeIfPresent([Token].self, forKey: .additionalSpecialTokens) ?? []
      if let templates = try? container.decodeIfPresent([ChatTemplate].self, forKey: .chatTemplate) {
        self.chatTemplate = templates
      } else {
        self.chatTemplate =
          try container.decodeIfPresent(ChatTemplate.self, forKey: .chatTemplate).map { [$0] } ?? []
      }
    }
  }

#endif
