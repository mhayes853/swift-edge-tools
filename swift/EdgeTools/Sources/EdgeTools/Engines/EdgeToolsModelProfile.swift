#if XGrammar
  import EdgeToolsXGrammar
#endif

import EdgeToolsCore
import EdgeToolsTokenizers

// MARK: - EdgeToolsModelProfile

public protocol EdgeToolsModelProfile: SendableMetatype {
  associatedtype Prompt: Sendable
  associatedtype GenerateParameters: EdgeToolsEngineGenerateParameters
  associatedtype GenerationParser: EdgeToolsGenerationParser
  associatedtype GrammarEngine: EdgeToolsGrammarEngine

  static var extraStopTokens: Set<String> { get }

  static func grammarEngine(
    tokenizer: any EdgeToolsTokenizer,
    vocabularySize: Int,
    stopTokenIds: Set<EdgeToolsToken.ID>
  ) throws -> GrammarEngine

  static func grammar(
    prompt: Prompt,
    tools: [EdgeToolDefinition],
    parameters: GenerateParameters,
    grammarEngine: borrowing GrammarEngine
  ) throws -> GrammarEngine.Grammar

  static func prepare(
    prompt: inout Prompt,
    tools: [EdgeToolDefinition],
    parser: inout GenerationParser
  )

  static func templateContext(prompt: Prompt) -> [String: EdgeToolsValue]?

  static func defaultSampling(
    prompt: Prompt,
    parameters: GenerateParameters
  ) -> EdgeToolsFusedSamplingParameters?
}

extension EdgeToolsModelProfile
where
  GenerateParameters: EdgeToolsConstrainedGenerateParameters,
  GenerateParameters.Constraint.Grammar == GrammarEngine.Grammar,
  GenerateParameters.Constraint.Context == GrammarEngine
{
  public static func constrainedGrammar(
    tools: [EdgeToolDefinition],
    parameters: GenerateParameters,
    grammarEngine: borrowing GrammarEngine,
    toolCallGrammar: (GrammarToolCallRange) throws -> GrammarEngine.Grammar
  ) throws -> GrammarEngine.Grammar {
    let constraint = parameters.constraint
    let grammar = try constraint.toolCallRange.map(toolCallGrammar)
    return try constraint.grammar(toolCallGrammar: grammar, context: grammarEngine)
  }
}

extension EdgeToolsModelProfile {
  public static var extraStopTokens: Set<String> { [] }

  public static func defaultSampling(
    prompt: Prompt,
    parameters: GenerateParameters
  ) -> EdgeToolsFusedSamplingParameters? {
    nil
  }

  public static func prepare(
    prompt: inout Prompt,
    tools: [EdgeToolDefinition],
    parser: inout GenerationParser
  ) {}

  public static func templateContext(prompt: Prompt) -> [String: EdgeToolsValue]? {
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
