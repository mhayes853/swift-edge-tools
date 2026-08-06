import EdgeTools
import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - GenericLLMMLXModel

/// Adapts an arbitrary MLX language model loaded by `LLMModelFactory` so unrecognized
/// architectures can still be run, without a model-specific tool call grammar or parser.
final class GenericLLMMLXModel: Module, LLMModel, EdgeToolsMLXModel {
  typealias ModelConfiguration = GenericLLMConfiguration
  typealias Prompt = EdgeToolsLLMPrompt
  typealias GenerateParameters = DefaultEdgeToolsMLXGenerateParameters
  typealias ToolCallParser = UnparsedToolCallParser
  typealias GrammarCompiler = XGRCompiler
  typealias GrammarContext = XGRGrammarContext

  let base: any LanguageModel
  let vocabularySize: Int

  var loraLayers: [Module] { [] }

  init(base: any LanguageModel, vocabularySize: Int) {
    self.base = base
    self.vocabularySize = vocabularySize
  }

  func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
    try self.base.prepare(input, cache: cache, windowSize: windowSize)
  }

  func callAsFunction(
    _ input: LMInput.Text,
    cache: [KVCache]?,
    state: LMOutput.State?
  ) -> LMOutput {
    self.base.callAsFunction(input, cache: cache, state: state)
  }

  func newCache(parameters: MLXLMCommon.GenerateParameters?) -> [KVCache] {
    self.base.newCache(parameters: parameters)
  }

  func toolCallGrammar(
    tools _: [EdgeToolDefinition],
    range _: GrammarToolCallRange
  ) throws -> XGRGrammar {
    throw EdgeCLIError(
      """
      This architecture has no tool call grammar in EdgeTools. Run with `--grammar unconstrained` \
      or supply your own grammar file.
      """
    )
  }
}

// MARK: - GenericLLMConfiguration

/// `GenericLLMMLXModel` is built from an already-loaded model rather than from a configuration
/// file, so no configuration fields are read.
struct GenericLLMConfiguration: Decodable {}

// MARK: - UnparsedToolCallParser

/// Emits nothing: an unrecognized architecture has no known tool call syntax to parse.
struct UnparsedToolCallParser: EdgeToolCallParser {
  mutating func accept(token _: EdgeToolsToken) -> EdgeRawToolCall? { nil }
}

// MARK: - Loading

func makeGenericLLMRunner(from directory: URL) async throws -> any EdgeRunner {
  let context = try await LLMModelFactory.shared.load(
    from: directory,
    using: #huggingFaceTokenizerLoader()
  )
  guard
    let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
      as? PreTrainedTokenizer
  else {
    throw EdgeCLIError("\(directory.lastPathComponent) has no supported tokenizer.")
  }
  let engine = try EdgeToolsMLXEngine<GenericLLMMLXModel>(
    model: GenericLLMMLXModel(
      base: context.model,
      vocabularySize: try vocabularySize(in: directory)
    ),
    tokenizer: EdgeToolsPreTrainedTokenizer(
      tokenizer: tokenizer,
      backendJSON: try huggingFaceBackendJSON(in: directory)
    )
  )
  return EngineRunner(
    engine: engine,
    clearCaches: { await engine.clearCaches() },
    supportsCustomGrammar: true,
    supportsSampling: true,
    makePrompt: llmPrompt,
    makeParameters: defaultMLXParameters
  )
}

private func vocabularySize(in directory: URL) throws -> Int {
  let url = directory.appending(path: "config.json")
  guard let data = try? Data(contentsOf: url),
    let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
    let size = json["vocab_size"] as? Int
  else {
    throw EdgeCLIError("config.json has no vocab_size, which is needed to constrain generation.")
  }
  return size
}

/// XGrammar needs the tokenizer's decoder, normalizer and pre-tokenizer to map grammar bitmasks
/// onto token ids.
private func huggingFaceBackendJSON(in directory: URL) throws -> String {
  let url = directory.appending(path: "tokenizer.json")
  guard let data = try? Data(contentsOf: url),
    let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
  else {
    throw EdgeCLIError("Could not read \(url.lastPathComponent).")
  }
  let fields = ["decoder", "normalizer", "pre_tokenizer"]
    .reduce(into: [String: Any]()) {
      $0[$1] = json[$1] ?? NSNull()
    }
  let encoded = try JSONSerialization.data(withJSONObject: fields)
  return String(decoding: encoded, as: UTF8.self)
}
