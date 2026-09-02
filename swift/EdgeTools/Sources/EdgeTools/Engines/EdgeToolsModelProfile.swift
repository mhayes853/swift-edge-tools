#if XGrammar
  import EdgeToolsXGrammar
#endif

import EdgeToolsCore
import EdgeToolsTokenizers

// MARK: - EdgeToolsModelProfile

public protocol EdgeToolsModelProfile: SendableMetatype {
  associatedtype Prompt: Sendable
  associatedtype GenerationParser: EdgeToolsGenerationParser
  associatedtype GrammarEngine: EdgeToolsGrammarEngine
  associatedtype Constraint: EdgeToolsGenerationConstraint
  where Constraint.Grammar == GrammarEngine.Grammar, Constraint.Context == GrammarEngine

  static var extraStopTokens: Set<String> { get }

  static func grammarEngine(
    tokenizer: any EdgeToolsTokenizer,
    vocabularySize: Int,
    stopTokenIds: Set<EdgeToolsToken.ID>
  ) throws -> GrammarEngine

  static func grammar(
    prompt: Prompt,
    reasoningEffort: EdgeToolsReasoningEffort,
    tools: [EdgeToolDefinition],
    constraint: Constraint,
    grammarEngine: borrowing GrammarEngine
  ) throws -> GrammarEngine.Grammar

  static func prepare(
    prompt: inout Prompt,
    reasoningEffort: EdgeToolsReasoningEffort,
    tools: [EdgeToolDefinition],
    parser: inout GenerationParser
  )

  static func templateContext(
    prompt: Prompt,
    reasoningEffort: EdgeToolsReasoningEffort
  ) -> [String: EdgeToolsValue]?

  static func defaultSampling(
    prompt: Prompt,
    reasoningEffort: EdgeToolsReasoningEffort
  ) -> EdgeToolsFusedSamplingParameters?
}

// MARK: - EdgeToolsMultimodalModelProfile

@nonexhaustive
public enum EdgeToolsMultimodalContent: Hashable, Sendable {
  case text(String)
  case image(EdgeToolsTranscript.Asset)
  case video(EdgeToolsTranscript.Asset)
  case audio(EdgeToolsTranscript.Asset)
}

public protocol EdgeToolsMultimodalModelProfile: EdgeToolsModelProfile
where Prompt == EdgeToolsTranscript {
  static func multimodalContent(
    for message: EdgeToolsTranscript.UserMessage
  ) -> [EdgeToolsMultimodalContent]
}

extension EdgeToolsMultimodalModelProfile {
  public static func multimodalContent(
    for message: EdgeToolsTranscript.UserMessage
  ) -> [EdgeToolsMultimodalContent] {
    message.images.map(EdgeToolsMultimodalContent.image)
      + message.videos.map(EdgeToolsMultimodalContent.video)
      + message.audio.map(EdgeToolsMultimodalContent.audio)
      + [.text(message.content)]
  }
}

extension EdgeToolsModelProfile {
  public static func constrainedGrammar(
    tools: [EdgeToolDefinition],
    constraint: Constraint,
    grammarEngine: borrowing GrammarEngine,
    toolCallGrammar: (GrammarToolCallRange) throws -> GrammarEngine.Grammar
  ) throws -> GrammarEngine.Grammar {
    let grammar = try (!tools.isEmpty ? constraint.toolCallRange : nil).map(toolCallGrammar)
    return try constraint.grammar(toolCallGrammar: grammar, context: grammarEngine)
  }
}

extension EdgeToolsModelProfile {
  public static var extraStopTokens: Set<String> { [] }

  public static func defaultSampling(
    prompt: Prompt,
    reasoningEffort: EdgeToolsReasoningEffort
  ) -> EdgeToolsFusedSamplingParameters? {
    nil
  }

  public static func prepare(
    prompt: inout Prompt,
    reasoningEffort: EdgeToolsReasoningEffort,
    tools: [EdgeToolDefinition],
    parser: inout GenerationParser
  ) {}

  public static func templateContext(
    prompt: Prompt,
    reasoningEffort: EdgeToolsReasoningEffort
  ) -> [String: EdgeToolsValue]? {
    nil
  }
}

#if XGrammar
  // MARK: - EdgeToolsModelProfile + XGrammar

  extension EdgeToolsModelProfile where GrammarEngine == XGrammarEngine {
    public static func grammarEngine(
      tokenizer: any EdgeToolsTokenizer,
      vocabularySize: Int,
      stopTokenIds: Set<EdgeToolsToken.ID>
    ) throws -> XGrammarEngine {
      guard let tokenizer = tokenizer as? any XGRTokenizer else {
        throw EdgeToolsError.unsupportedTokenizer
      }
      return try XGrammarEngine(
        tokenizerInfo: tokenizer.tokenizerInfo(
          modelVocabularySize: vocabularySize,
          extraStopTokenIds: stopTokenIds
        )
      )
    }
  }
#endif
