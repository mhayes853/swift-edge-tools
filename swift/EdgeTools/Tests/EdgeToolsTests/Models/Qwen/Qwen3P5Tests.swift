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
        func `Forked Image Prefill Only Processes Text Suffix`() async throws {
          let engine = try await Qwen3P5VLMLXModelEngine(from: downloadQwen3P5VL())
          let parameters = MLXContextParameters(
            transcript: EdgeToolsTranscript(
              messages: [
                .user("Use this image as a reference.", images: [try redImageAsset()])
              ],
              reasoningEffort: .none
            )
          )
          let context = engine.context(parameters)
          let initial = try await engine.prefill(context: context)
          let prompt = EdgeToolsTranscript.UserMessage("Answer red or blue.")

          let forked = try await qwenVLMGeneration(
            using: engine,
            prompt: prompt,
            context: context.fork()
          )
          let fresh = try await qwenVLMGeneration(
            using: engine,
            prompt: prompt,
            context: engine.context(parameters)
          )
          let forkedTokenIDs = forked.tokens.map(\.id)
          let freshTokenIDs = fresh.tokens.map(\.id)

          expectNoDifference(forkedTokenIDs, freshTokenIDs)
          let forkedPrefillTokens = forked.prefillMetrics.tokens + initial.metrics.tokens
          expectNoDifference(forkedPrefillTokens, fresh.prefillMetrics.tokens)
        }

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

  #if HuggingFaceTokenizers && Llama && XGrammar && canImport(CLlama) && !os(WASI)
    @Suite(.serialized)
    struct `Qwen3P5LlamaModelEngine tests` {
      @Test
      func `Llama Hybrid Fork Falls Back To A Cold Cache`() async throws {
        let engine = try Qwen3P5LlamaModelEngine(
          modelPath: (try await downloadGGUFModel(id: .qwen3P5)).path()
        )
        let parameters = EdgeToolsTranscriptContextParameters(
          transcript: EdgeToolsTranscript(messages: [
            .user("Say hello in one word."),
            .assistant([.text("Hello.")])
          ]),
          reasoningEffort: .none
        )
        let context = engine.context(parameters)
        _ = try await engine.prefill(context: context)

        let forked = try await qwenLlamaGeneration(
          using: engine,
          prompt: .user("Now say goodbye in one word."),
          context: context.fork()
        )
        let fresh = try await qwenLlamaGeneration(
          using: engine,
          prompt: .user("Now say goodbye in one word."),
          context: engine.context(parameters)
        )

        expectNoDifference(forked.tokens, fresh.tokens)
        expectNoDifference(forked.prefillMetrics.tokens, fresh.prefillMetrics.tokens)
      }

      @Test
      func `Llama Image Fork Falls Back To A Cold Cache`() async throws {
        let engine = try await self.multimodalEngine()
        let parameters = EdgeToolsTranscriptContextParameters(
          transcript: try qwenLlamaImagePrefix(),
          reasoningEffort: .none
        )
        let context = engine.context(parameters)
        _ = try await engine.prefill(context: context)

        let forked = try await qwenLlamaGeneration(
          using: engine,
          prompt: .user("Answer red or blue."),
          context: context.fork()
        )
        let fresh = try await qwenLlamaGeneration(
          using: engine,
          prompt: .user("Answer red or blue."),
          context: engine.context(parameters)
        )

        expectNoDifference(forked.tokens, fresh.tokens)
        expectNoDifference(forked.prefillMetrics.tokens, fresh.prefillMetrics.tokens)
      }

      @Test
      func `Llama Image Fork With Another Image Matches A Cold Cache`() async throws {
        let engine = try await self.multimodalEngine()
        let parameters = EdgeToolsTranscriptContextParameters(
          transcript: try qwenLlamaImagePrefix(),
          reasoningEffort: .none
        )
        let context = engine.context(parameters)
        _ = try await engine.prefill(context: context)
        let prompt = EdgeToolsTranscript.UserMessage(
          "Is this image also red?",
          images: [try llamaRedImageAsset()]
        )

        let forked = try await qwenLlamaGeneration(
          using: engine,
          prompt: prompt,
          context: context.fork()
        )
        let fresh = try await qwenLlamaGeneration(
          using: engine,
          prompt: prompt,
          context: engine.context(parameters)
        )

        expectNoDifference(forked.tokens, fresh.tokens)
        expectNoDifference(forked.prefillMetrics.tokens, fresh.prefillMetrics.tokens)
      }

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

      private func multimodalEngine() async throws -> Qwen3P5LlamaModelEngine {
        let model = try await downloadGGUFMultimodalModel(id: .qwen3P5VL)
        return try Qwen3P5LlamaModelEngine(
          modelPath: model.model.path(),
          multimodalProjectorPath: model.projector.path(),
          contextParameters: LlamaContextParameters(maximumSequenceCount: 2),
          multimodalParameters: LlamaMultimodalParameters(warmUp: false)
        )
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

#if MLX && XGrammar && canImport(MLX) && canImport(CoreImage) && canImport(MLXVLM) && !os(WASI)
  private func qwenVLMGeneration(
    using engine: Qwen3P5VLMLXModelEngine,
    prompt: EdgeToolsTranscript.UserMessage,
    context: MLXContext<Qwen3P5VLMLXProfile>
  ) async throws -> EdgeToolsEngineGeneration {
    let task = try engine.generate(
      prompt: .user(prompt),
      tools: [],
      parameters: DefaultMLXGenerateParameters(
        sampling: .greedy,
        maxTokens: 1,
        synchronizeStreamForMemorySnapshots: false
      ),
      context: context,
      channel: EdgeToolsGenerationChannel()
    )
    return try await task.value
  }
#endif

#if HuggingFaceTokenizers && Llama && XGrammar && canImport(CLlama) && !os(WASI)
  private func qwenLlamaImagePrefix() throws -> EdgeToolsTranscript {
    EdgeToolsTranscript(
      messages: [
        .user("What is the dominant color?", images: [try llamaRedImageAsset()]),
        .assistant([.text("The dominant color is red.")])
      ],
      reasoningEffort: .none
    )
  }

  private func qwenLlamaGeneration(
    using engine: Qwen3P5LlamaModelEngine,
    prompt: EdgeToolsTranscript.UserMessage,
    context: LlamaContext<Qwen3P5LlamaProfile>
  ) async throws -> EdgeToolsEngineGeneration {
    let task = try engine.generate(
      prompt: .user(prompt),
      tools: [],
      parameters: DefaultLlamaGenerateParameters(sampling: .greedy, maxTokens: 1),
      context: context,
      channel: EdgeToolsGenerationChannel()
    )
    return try await task.value
  }
#endif
