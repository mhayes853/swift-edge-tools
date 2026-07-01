#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI
  import Tokenizers
  import Foundation

  @available(anyAppleOS 27.0, *)
  public final class NeedleCoreAIEngine: NeedleEngine, Sendable {
    public final class GenerationTask: NeedleEngineGenerationTask {
      public var value: NeedleEngineGeneration {
        get async throws { .empty }
      }

      public func stop() {}
    }

    public struct GenerateParameters: NeedleEngineGenerateParameters {
      public static var `default`: Self { Self() }
    }

    private struct State {
      let grammarEngine: NeedleXGrammarEngine
    }

    private let state: Lock<State>
    private let model: AIModel
    private let tokenizer: any Tokenizer

    public convenience init(modelDirectoryURL: URL) throws {
      fatalError()
    }

    public init(
      model: AIModel,
      tokenizer: any Tokenizer,
      grammarEngine: sending NeedleXGrammarEngine
    ) throws {
      self.state = Lock(State(grammarEngine: grammarEngine))
      self.model = model
      self.tokenizer = tokenizer
    }

    public func tokenize(prompt: NeedlePrompt) -> [NeedleToken] {
      prompt.tokenized(using: self.tokenizer)
    }

    public func clearCaches() {
      self.state.withLock { $0.grammarEngine.clearCache() }
    }

    public func generate(
      prompt: NeedlePrompt,
      parameters: sending GenerateParameters,
      onToken: @escaping @Sendable (NeedleToken) -> Void
    ) throws -> GenerationTask {
      GenerationTask()
    }
  }
#endif
