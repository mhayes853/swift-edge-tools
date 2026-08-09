#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && XGrammar && canImport(CoreImage) && canImport(MLX) && canImport(MLXVLM)
  import Foundation
  import MLXLMCommon
  import MLXNN
  import MLXVLM

  public struct Gemma4MLXProfile: MLXVLMModelProfile {
    public typealias Prompt = EdgeToolsLLMPrompt
    public typealias GenerationParser = Gemma4GenerationParser
    public typealias GenerateParameters = DefaultMLXGenerateParameters
    public typealias GrammarCompiler = XGRCompiler
    public typealias GrammarContext = XGRGrammarContext

    public static var extraStopTokens: Set<String> { ["<|tool_response>"] }

    public static func grammar(
      prompt: EdgeToolsLLMPrompt,
      tools: [EdgeToolDefinition],
      parameters: DefaultMLXGenerateParameters,
      context: XGRGrammarContext
    ) throws -> XGRGrammar {
      try Self.constrainedGrammar(tools: tools, parameters: parameters, context: context) { range in
        let toolCalls = try XGRGrammar.gemma4(tools: tools, range: range)
        guard prompt.reasoningEffort != .default, prompt.reasoningEffort.isEnabled else {
          return toolCalls
        }
        return try XGRGrammar.gemma4Reasoning().concatenate(toolCalls)
      }
    }

    public static func prepare(
      prompt: inout EdgeToolsLLMPrompt,
      tools _: [EdgeToolDefinition],
      parser _: inout Gemma4GenerationParser
    ) {
      prompt = prompt.gemma4PreparedForReasoning
    }

    public static nonisolated(nonsending) func input(
      prompt: EdgeToolsLLMPrompt,
      tools: [EdgeToolDefinition],
      tokenizer _: any EdgeToolsTokenizer,
      processor: (any UserInputProcessor)?
    ) async throws -> LMInput {
      guard let processor else { throw EdgeToolsError.failedToLoadConfiguration }
      return try await processor.prepare(input: try prompt.gemma4UserInput(tools: tools))
    }
  }

  public typealias Gemma4MLXModelEngine = MLXEngine<Gemma4MLXProfile>

  extension EdgeToolsModelEngine where Model == EdgeToolsMLXModel<Gemma4MLXProfile> {
    public init(from directoryURL: URL) async throws {
      try await self.init(from: MLXModelDirectory(url: directoryURL))
    }

    public init(from directory: MLXModelDirectory) async throws {
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

  extension EdgeToolsLLMPrompt {
    fileprivate var gemma4PreparedForReasoning: Self {
      guard self.reasoningEffort != .default, self.reasoningEffort.isEnabled else { return self }
      var prompt = self
      if case .system(let instruction) = prompt.messages.first {
        guard !instruction.hasPrefix("<|think|>\n") else { return prompt }
        prompt.messages[0] = .system("<|think|>\n\(instruction)")
      } else {
        prompt.messages.insert(.system("<|think|>"), at: 0)
      }
      return prompt
    }

    fileprivate func gemma4UserInput(tools: [EdgeToolDefinition]) throws -> UserInput {
      try self.gemma4PreparedForReasoning.mlxUserInput(tools: tools) { message in
        switch message {
        case .system:
          return try message.mlxMessage()
        case .user(let text, let messageImages, videos: _, audio: _):
          var content: [MLXLMCommon.Message] = messageImages.map { _ in ["type": "image"] }
          content.append(["type": "text", "text": text])
          return ["role": "user", "content": content]
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

  extension XGRGrammar {
    static func gemma4Reasoning() throws -> XGRGrammar {
      let opener = try Self.literal("<|channel>thought\n")
      let thought = try opener.concatenate(.universal)
      return try thought.concatenate(Self.literal("<channel|>"))
    }
  }
#endif
