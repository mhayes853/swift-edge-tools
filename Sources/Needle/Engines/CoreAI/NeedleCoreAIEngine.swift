#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI
  import Tokenizers
  import Foundation

  @available(anyAppleOS 27.0, *)
  public final class NeedleCoreAIEngine: NeedleEngine {
    public struct GenerateParameters: NeedleEngineGenerateParameters {
      public static var `default`: Self { Self() }
    }

    public var stopper: NeedleEngineStopper {
      NeedleEngineStopper {}
    }

    public let model: AIModel
    public let tokenizer: any Tokenizer
    public let grammarEngine: NeedleXGrammarEngine

    public convenience init(modelDirectoryURL: URL) throws {
      fatalError()
    }

    public init(
      model: AIModel,
      tokenizer: any Tokenizer,
      grammarEngine: NeedleXGrammarEngine
    ) throws {
      self.model = model
      self.tokenizer = tokenizer
      self.grammarEngine = grammarEngine
    }

    public func tokenize(prompt: NeedlePrompt) -> [NeedleToken] {
      prompt.tokenized(using: self.tokenizer)
    }

    public func generate(
      prompt: NeedlePrompt,
      parameters: GenerateParameters,
      onToken: (NeedleToken) -> Void
    ) throws -> NeedleEngineGeneration {
      .empty
    }

    public func clearCaches() {
      self.grammarEngine.clearCache()
    }

    public func reset() {
      self.clearCaches()
    }
  }
#endif
