import EdgeTools
import Testing

#if !os(WASI)
  import SnapshotTesting
#endif

@Suite
struct `Qwen3P5 tests` {
  #if MLX && XGrammar && canImport(MLX) && !os(WASI)
    @Test
    func `Default Sampling Follows The Reasoning Effort`() {
      let thinking = Qwen3P5MLXProfile.defaultSampling(
        prompt: EdgeToolsTranscript(messages: [.user("hi")], reasoningEffort: .high),
        parameters: DefaultMLXGenerateParameters()
      )
      let nonThinking = Qwen3P5MLXProfile.defaultSampling(
        prompt: EdgeToolsTranscript(messages: [.user("hi")], reasoningEffort: .none),
        parameters: DefaultMLXGenerateParameters()
      )

      expectNoDifference(thinking?.topP, 0.95)
      expectNoDifference(thinking?.presencePenalty, 1.5)
      expectNoDifference(nonThinking?.topP, nil)
      expectNoDifference(nonThinking?.presencePenalty, 2)
    }

  #endif

  #if MLX && XGrammar && canImport(MLX) && !os(WASI)
    @Suite(.serialized, .enabledIfMLXTests())
    struct `Qwen3P5MLXModelEngine tests` {
      @Test
      func `Completes Tool Turn Snapshot`() async throws {
        let engine = try await Qwen3P5MLXModelEngine(from: downloadQwen3P5())
        let transcript = try await completeWeatherTurn(using: engine)

        withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
      }

      @Test
      func `Completes A Session Tool Turn With The Fused Sampler Snapshot`() async throws {
        let engine = try await Qwen3P5MLXModelEngine(from: downloadQwen3P5())
        let session = EdgeToolsSession(engine: engine) { WeatherTestTool() }
        let turn = try await completeWeatherTurn(
          using: session,
          sampling: EdgeToolsFusedSamplingParameters(
            temperature: 0.7,
            topK: 40,
            topP: 0.9,
            minP: 0.05,
            repetitionPenalty: 1.1,
            seed: 1234
          )
        )

        withKnownIssue { assertSnapshot(of: turn, as: .dump, record: .all) }
      }

      @Test
      func `Generates Reasoning Snapshot`() async throws {
        let engine = try await Qwen3P5MLXModelEngine(from: downloadQwen3P5())
        let generation = try await generateReasoning(using: engine)

        withKnownIssue { assertSnapshot(of: generation, as: .dump, record: .all) }
      }
    }

    #if canImport(CoreImage) && canImport(MLXVLM)
      @Suite(.serialized, .enabledIfMLXTests())
      struct `Qwen3P5VLMLXModelEngine tests` {
        @Test
        func `Describes Video Snapshot`() async throws {
          let engine = try await Qwen3P5VLMLXModelEngine(from: downloadQwen3P5VL())
          let response = try await describeRedVideo(using: engine)

          withKnownIssue { assertSnapshot(of: response, as: .lines, record: .all) }
        }

        @Test
        func `Describes Image And Video Snapshot`() async throws {
          let engine = try await Qwen3P5VLMLXModelEngine(from: downloadQwen3P5VL())
          let response = try await describeRedImageAndVideo(using: engine)

          withKnownIssue { assertSnapshot(of: response, as: .lines, record: .all) }
        }

        @Test
        func `Completes Video Conditioned Tool Turn Snapshot`() async throws {
          let engine = try await Qwen3P5VLMLXModelEngine(from: downloadQwen3P5VL())
          let result = try await completeVideoColorTurn(using: engine)

          withKnownIssue { assertSnapshot(of: result, as: .dump, record: .all) }
        }

        @Test
        func `Generates Reasoning Snapshot`() async throws {
          let engine = try await Qwen3P5VLMLXModelEngine(from: downloadQwen3P5VL())
          let generation = try await generateReasoning(using: engine)

          withKnownIssue { assertSnapshot(of: generation, as: .dump, record: .all) }
        }
      }
    #endif
  #endif

  #if Llama && XGrammar && canImport(CLlama) && !os(WASI)
    @Suite(.serialized)
    struct `Qwen3P5LlamaModelEngine tests` {
      @Test
      func `Llama Completes Tool Turn Snapshot`() async throws {
        let engine = try Qwen3P5LlamaModelEngine(
          modelPath: (try await downloadGGUFModel(id: .qwen3P5)).path()
        )
        let transcript = try await completeWeatherTurn(using: engine)

        withKnownIssue { assertSnapshot(of: transcript, as: .dump, record: .all) }
      }

      @Test
      func `Llama Generates Reasoning Snapshot`() async throws {
        let engine = try Qwen3P5LlamaModelEngine(
          modelPath: (try await downloadGGUFModel(id: .qwen3P5)).path()
        )
        let generation = try await generateReasoning(using: engine)

        withKnownIssue { assertSnapshot(of: generation, as: .dump, record: .all) }
      }
    }

    @Suite(.serialized)
    struct `Qwen3P5VLLlamaModelEngine tests` {
      @Test
      func `Llama Describes Image Snapshot`() async throws {
        let engine = try await self.multimodalEngine()
        let response = try await describeRedImage(using: engine)

        withKnownIssue { assertSnapshot(of: response, as: .lines, record: .all) }
      }

      @Test
      func `Llama Completes Image Conditioned Tool Turn Snapshot`() async throws {
        let engine = try await self.multimodalEngine()
        let result = try await completeImageColorTurn(using: engine)

        withKnownIssue { assertSnapshot(of: result, as: .dump, record: .all) }
      }

      private func multimodalEngine() async throws -> Qwen3P5VLLlamaModelEngine {
        let model = try await downloadGGUFMultimodalModel(id: .qwen3P5VL)
        return try Qwen3P5VLLlamaModelEngine(
          modelPath: model.model.path(),
          multimodalProjectorPath: model.projector.path(),
          contextParameters: LlamaContextParameters(maximumSequenceCount: 1),
          multimodalParameters: LlamaMultimodalParameters(warmUp: false)
        )
      }
    }
  #endif
}
