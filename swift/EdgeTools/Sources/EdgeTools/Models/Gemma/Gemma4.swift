#if XGrammar
  import EdgeToolsCore
  import EdgeToolsXGrammar

  // MARK: - Gemma4 Model

  public struct Gemma4Profile: EdgeToolsMultimodalModelProfile {
    public typealias GenerationParser = Gemma4GenerationParser
    public typealias GrammarEngine = XGrammarEngine
    public typealias Constraint = XGRGenerationConstraint

    public static var extraStopTokens: Set<String> { ["<|tool_response>"] }

    public static func grammar(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      constraint: XGRGenerationConstraint,
      grammarEngine: borrowing XGrammarEngine
    ) throws -> XGRGrammar {
      let grammar = try Self.constrainedGrammar(
        tools: tools,
        constraint: constraint,
        grammarEngine: grammarEngine
      ) {
        try XGRGrammar.gemma4(tools: tools, range: $0)
      }
      guard reasoningEffort != .default, reasoningEffort.isEnabled else {
        return grammar
      }
      return try XGRGrammar.gemma4Reasoning().concatenate(grammar)
    }

    public static func prepare(
      prompt: inout EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      parser: inout Gemma4GenerationParser
    ) {
      prompt = prompt.gemma4PreparedForReasoning(reasoningEffort: reasoningEffort)
    }
  }
#endif

#if MLX && canImport(CoreImage) && canImport(MLX) && canImport(MLXVLM)
  import EdgeToolsTokenizers
  import Foundation
  import MLXLMCommon
  import MLXNN
  import MLXVLM

  // MARK: - Gemma4 MLX Model

  extension Gemma4Profile: MLXVLMModelProfile {
    public static nonisolated(nonsending) func input(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput {
      guard let processor else { throw EdgeToolsError.failedToLoadConfiguration }
      return try await processor.prepare(
        input: try prompt.gemma4UserInput(
          reasoningEffort: reasoningEffort,
          tools: tools,
          addGenerationPrompt: true
        )
      )
    }

    public static nonisolated(nonsending) func prefillInput(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      tokenizer: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput {
      guard let processor else { throw EdgeToolsError.failedToLoadConfiguration }
      return try await processor.prepare(
        input: try prompt.gemma4UserInput(
          reasoningEffort: reasoningEffort,
          tools: tools,
          addGenerationPrompt: false
        )
      )
    }
  }

  public typealias Gemma4MLXModelEngine = MLXEngine<Gemma4Profile>

  extension MLXEngine where Profile == Gemma4Profile {
    public convenience init(from directoryURL: URL) async throws {
      try await self.init(from: MLXModelDirectory(url: directoryURL))
    }

    public convenience init(from directory: MLXModelDirectory) async throws {
      try await self.init(from: directory) { weights, model in
        guard let model = model as? MLXVLM.Gemma4 else { return }
        let firstSharedLayer =
          model.config.textConfiguration.hiddenLayers
          - model.config.textConfiguration.numKVSharedLayers
        guard firstSharedLayer > 0 else { return }
        for (key, value) in model.parameters().flattened() where weights[key] == nil {
          let components = key.split(separator: ".")
          guard
            components.count > 6,
            components[0] == "language_model",
            components[1] == "model",
            components[2] == "layers",
            let layer = Int(components[3]),
            layer >= firstSharedLayer,
            components[4] == "self_attn",
            components[5] == "k_proj" || components[5] == "v_proj" || components[5] == "k_norm"
          else { continue }
          weights[key] = value
        }
      }
    }
  }
#endif

#if Llama && canImport(CLlama)
  // MARK: - Gemma4 Llama Model

  extension Gemma4Profile: LlamaModelProfile {}

  public typealias Gemma4LlamaModelEngine = LlamaEngine<Gemma4Profile>
#endif

// MARK: - Gemma4 Grammar

#if XGrammar
  extension XGRGrammar {
    static func gemma4Reasoning() throws -> XGRGrammar {
      let opener = try Self.literal("<|channel>thought\n")
      let thought = try opener.concatenate(.universal)
      return try thought.concatenate(Self.literal("<channel|>"))
    }
  }
#endif

// MARK: - Gemma4 Reasoning Preparation

extension EdgeToolsTranscript {
  func gemma4PreparedForReasoning(
    reasoningEffort: EdgeToolsReasoningEffort
  ) -> Self {
    guard reasoningEffort != .default, reasoningEffort.isEnabled else { return self }
    var prompt = self
    if case .system(let message) = prompt.messages.first {
      guard !message.content.hasPrefix("<|think|>\n") else { return prompt }
      prompt.messages[0] = .system(
        EdgeToolsTranscript.SystemMessage(content: "<|think|>\n\(message.content)")
      )
    } else {
      prompt.messages.insert(
        .system(EdgeToolsTranscript.SystemMessage(content: "<|think|>")),
        at: 0
      )
    }
    return prompt
  }
}

// MARK: - Gemma4 MLX Input

#if MLX && canImport(CoreImage) && canImport(MLX) && canImport(MLXVLM)
  extension EdgeToolsTranscript {
    fileprivate func gemma4UserInput(
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      addGenerationPrompt: Bool
    ) throws -> UserInput {
      try self.gemma4PreparedForReasoning(reasoningEffort: reasoningEffort).mlxUserInput(
        tools: tools,
        additionalContext: ["add_generation_prompt": .boolean(addGenerationPrompt)]
      ) { message in
        switch message {
        case .system:
          return try message.mlxMessage()
        case .user(let message):
          return [
            "role": "user",
            "content": Gemma4Profile.multimodalContent(for: message).map(\.mlxMessage)
          ]
        case .assistant, .tool:
          var result = try message.mlxMessage()
          if let text = result["content"] as? String {
            result["content"] = [["type": "text", "text": text]] as [MLXLMCommon.Message]
          }
          return result
        }
      }
    }
  }
#endif
