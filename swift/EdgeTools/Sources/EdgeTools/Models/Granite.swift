#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && canImport(MLX)
  import EdgeToolsCore

  // MARK: - Granite Model

  public struct GraniteMLXProfile: MLXLLMModelProfile {
    public typealias Prompt = EdgeToolsTranscript
    public typealias GenerationParser = GraniteGenerationParser
    public typealias GrammarEngine = XGrammarEngine

    public static func grammar(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      parameters: MLXGenerateParameters,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(tools: tools, parameters: parameters, grammarEngine: grammarEngine) {
        try .granite(tools: tools, range: $0)
      }
    }
  }

  public typealias GraniteMLXModelEngine = MLXEngine<GraniteMLXProfile>

  // MARK: - GraniteMoeHybrid Model

  public struct GraniteMoeHybridMLXProfile: MLXLLMModelProfile {
    public typealias Prompt = EdgeToolsTranscript
    public typealias GenerationParser = GraniteGenerationParser
    public typealias GrammarEngine = XGrammarEngine

    public static func grammar(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      parameters: MLXGenerateParameters,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(tools: tools, parameters: parameters, grammarEngine: grammarEngine) {
        try XGRGrammar.granite(tools: tools, range: $0)
      }
    }
  }

  public typealias GraniteMoeHybridMLXModelEngine = MLXEngine<GraniteMoeHybridMLXProfile>
#endif

#if Llama && canImport(CLlama)
  import EdgeToolsCore

  public struct GraniteLlamaProfile: LlamaModelProfile {
    public typealias Prompt = EdgeToolsTranscript
    public typealias GenerationParser = GraniteGenerationParser
    public typealias GrammarEngine = XGrammarEngine

    public static func grammar(
      prompt: EdgeToolsTranscript,
      tools: [EdgeToolDefinition],
      parameters: LlamaGenerateParameters,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(tools: tools, parameters: parameters, grammarEngine: grammarEngine) {
        try .granite(tools: tools, range: $0)
      }
    }
  }

  public typealias GraniteLlamaModelEngine = LlamaEngine<GraniteLlamaProfile>
#endif

// MARK: - Granite Tool Call Parsing

public struct GraniteGenerationParser: EdgeToolsGenerationParser, Sendable {
  private var base = DelimitedGenerationParser(
    toolOpener: "<tool_call>",
    toolCloser: "</tool_call>",
    parseToolCalls: graniteToolCalls(in:)
  )

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> [EdgeToolsGenerationPart] {
    self.base.accept(token: token)
  }

  public mutating func finish() -> [EdgeToolsGenerationPart] {
    self.base.finish()
  }
}

private func graniteToolCalls(in source: String) -> [EdgeRawToolCall] {
  var block = IncrementalToolCallBlock(opener: "<tool_call>", closer: "</tool_call>")
  block.append(EdgeToolsToken(id: -1, stringValue: source))
  var calls = [EdgeRawToolCall]()
  while let payload = block.nextPayload(respectingJSONStringBoundaries: true) {
    if let value = try? EdgeToolsValue(json: payload), let call = EdgeRawToolCall(jsonValue: value)
    {
      calls.append(call)
    }
  }
  return calls
}

// MARK: - Granite Grammar

#if XGrammar
  extension XGRGrammar {
    public static func granite(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      try .qwenJSON(tools: tools, range: range)
    }
  }
#endif
