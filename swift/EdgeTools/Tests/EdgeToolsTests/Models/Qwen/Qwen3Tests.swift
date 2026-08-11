import EdgeTools
import Testing

#if !os(WASI)
  import SnapshotTesting
#endif

@Suite
struct `Qwen3 tests` {
  #if MLX && XGrammar && canImport(MLX) && !os(WASI)
    @Suite(.serialized, .enabledIfMLXTests())
    struct `Qwen3MLXModelEngine tests` {
      @Test
      func `Completes Tool Turn Snapshot`() async throws {
        let engine = try await Qwen3MLXModelEngine(from: downloadQwen3())
        let transcript = try await completeWeatherTurn(using: engine)

        withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
      }

      @Test
      func `Generates Reasoning Snapshot`() async throws {
        let engine = try await Qwen3MLXModelEngine(from: downloadQwen3())
        let generation = try await generateReasoning(using: engine)

        withKnownIssue { assertSnapshot(of: generation, as: .dump, record: .all) }
      }

      @Test
      func `Forked System Prefill Only Processes User Suffix`() async throws {
        let engine = try await Qwen3MLXModelEngine(from: downloadQwen3())
        let systemPrompt = "You are a helpful assistant."
        let context = engine.context(
          MLXContextParameters(
            transcript: EdgeToolsTranscript(messages: [.system(systemPrompt)]),
            reasoningEffort: .none
          )
        )
        let prefill = try await engine.prefill(context: context)

        let forkedTask = try engine.generate(
          prompt: .user("Say hello in one word."),
          tools: [],
          parameters: DefaultMLXGenerateParameters(
            maxTokens: 1,
            synchronizeStreamForMemorySnapshots: false
          ),
          context: context.fork(),
          channel: EdgeToolsGenerationChannel()
        )
        let forked = try await forkedTask.value

        let freshTask = try engine.generate(
          prompt: .user("Say hello in one word."),
          tools: [],
          parameters: DefaultMLXGenerateParameters(
            maxTokens: 1,
            synchronizeStreamForMemorySnapshots: false
          ),
          context: engine.context(
            MLXContextParameters(
              transcript: EdgeToolsTranscript(messages: [.system(systemPrompt)]),
              reasoningEffort: .none
            )
          ),
          channel: EdgeToolsGenerationChannel()
        )
        let fresh = try await freshTask.value

        expectNoDifference(
          forked.prefillMetrics.tokens + prefill.metrics.tokens,
          fresh.prefillMetrics.tokens
        )
      }
    }
  #endif
}
