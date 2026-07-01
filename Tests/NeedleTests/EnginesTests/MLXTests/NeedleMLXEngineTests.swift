#if MLX && XGrammar && canImport(MLX)
  import Needle
  import Testing
  import CustomDump
  import SnapshotTesting
  import IssueReporting
  import MLX
  import MLXLMCommon

  @Suite(.serialized, .enabledIfXcode())
  struct `NeedleMLXEngine tests` {
    private let engine: NeedleMLXEngine

    init() async throws {
      self.engine = try await NeedleMLXEngine(from: downloadNeedleHF())
    }

    @Test
    func `Generate Basics`() async throws {
      let tokens = Lock([NeedleToken]())
      let generationTask = try self.engine.generate(
        prompt: .sendAdventureEmail,
        parameters: .default
      ) { token in tokens.withLock { $0.append(token) } }
      let generation = try await generationTask.value
      expectNoDifference(generation.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.metadata, as: .dump, record: .all)
        assertSnapshot(of: generation.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    func `Generate Streamed Response Matches Final Response`() async throws {
      let tokens = Lock([NeedleToken]())
      let generationTask = try self.engine.generate(
        prompt: .sendAdventureEmail,
        parameters: .default
      ) { token in tokens.withLock { $0.append(token) } }
      let generation = try await generationTask.value

      let streamedResponse = tokens.withLock { $0.map(\.stringValue).joined() }
      let finalResponse = generation.tokens.map(\.stringValue).joined()
      expectNoDifference(streamedResponse, finalResponse)
    }

    @Test
    func `Generate Stops And Returns Stopped Generation`() async throws {
      let tokens = Lock([NeedleToken]())
      let generationTaskBox = Lock<NeedleMLXEngine.GenerationTask?>(nil)
      let generationTask = try self.engine.generate(
        prompt: .sendAdventureEmail,
        parameters: .default,
      ) { token in
        tokens.withLock { $0.append(token) }
        generationTaskBox.withLock { $0?.stop() }
      }
      generationTaskBox.withLock { $0 = generationTask }
      let generation = try await generationTask.value

      expectNoDifference(generation.wasStopped, true)
      let tokenCount = tokens.withLock { $0.count }
      expectNoDifference(tokenCount > 0, true)
      expectNoDifference(generation.decodeMetrics.tokens, tokenCount)
      expectNoDifference(generation.tokens.isEmpty, false)
    }

    @Test
    func `Generate Cancels And Throws Cancellation Error`() async throws {
      let task = Task {
        let generationTask = try self.engine.generate(
          prompt: .sendAdventureEmail,
          parameters: .default
        ) { _ in }
        _ = try await generationTask.value
      }

      task.cancel()
      await #expect(throws: CancellationError.self) {
        _ = try await task.value
      }
    }

    @Test
    func `Generate With 4 Bit KV Cache Quantization`() async throws {
      let tokenStorage = Lock([NeedleToken]())
      let generationTask = try self.engine.generate(
        prompt: .sendAdventureEmail,
        parameters: NeedleMLXEngine.GenerateParameters(kvCacheQuantizationBits: 4)
      ) { token in tokenStorage.withLock { $0.append(token) } }
      let generation = try await generationTask.value
      expectNoDifference(generation.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.metadata, as: .dump, record: .all)
        assertSnapshot(of: generation.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    func `Generate With Untied Word Embeddings`() async throws {
      let engine = try await NeedleMLXEngine(
        from: downloadNeedleHF(),
        editConfiguration: { configuration in
          configuration.tieWordEmbeddings = false
        }
      )

      let tokenStorage = Lock([NeedleToken]())
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        parameters: .default
      ) { token in tokenStorage.withLock { $0.append(token) } }
      let generation = try await generationTask.value
      expectNoDifference(generation.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.metadata, as: .dump, record: .all)
        assertSnapshot(of: generation.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    func `Generate Through NeedleSession`() async throws {
      let session = NeedleSession(engine: self.engine)
      let generation = try await session.generate(
        tools: [SendEmailTool()],
        with: NeedlePrompt.sendAdventureEmail.user
      )
      expectNoDifference(generation.engineGeneration.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(
          of: generation.engineGeneration.metadata,
          as: .dump,
          record: .all
        )
        assertSnapshot(
          of: generation.engineGeneration.tokens.map(\.stringValue).joined(),
          as: .lines,
          record: .all
        )
      }
    }

    @Test
    func `Generate Invokes Custom Logit Processor`() async throws {
      let processor = CountingLogitProcessor()

      let generationTask = try self.engine.generate(
        prompt: .sendAdventureEmail,
        parameters: NeedleMLXEngine.GenerateParameters(processor: processor)
      ) { _ in }
      _ = try await generationTask.value

      expectNoDifference(processor.promptCalls, 1)
      expectNoDifference(processor.processCalls > 0, true)
      expectNoDifference(processor.didSampleCalls, processor.processCalls)
    }

    @Test
    func `Generate Records Mean Confidence In Range Zero To One`() async throws {
      let generationTask = try self.engine.generate(
        prompt: .sendAdventureEmail,
        parameters: .default
      ) { _ in }
      let generation = try await generationTask.value

      let confidence = try #require(generation.metadata.mlxEngineGenerationConfidence)
      expectNoDifference((0...1).contains(confidence), true)

      let perToken = try #require(generation.metadata.mlxEnginePerTokenConfidences)
      expectNoDifference(perToken.count, generation.tokens.count)
      for uncertainty in perToken {
        expectNoDifference((0...1).contains(uncertainty), true)
      }
    }

    @Test
    func `Generate Throws When Prompt Exceeds Context Length`() async throws {
      let prompt = NeedlePrompt(
        system: "",
        user: String(repeating: "token ", count: 2_000),
        tools: [.sendEmail]
      )

      let error = await #expect(throws: NeedleMLXEngineError.self) {
        let generationTask = try self.engine.generate(prompt: prompt, parameters: .default) { _ in }
        _ = try await generationTask.value
      }
      expectNoDifference(error?.message.contains("context length"), true)
    }

    @Test
    func `Tokenize Base`() {
      assertSnapshot(of: self.engine.tokenize(prompt: .sendAdventureEmail), as: .dump)
    }
  }

  // MARK: - CountingLogitProcessor

  final class CountingLogitProcessor: LogitProcessor, @unchecked Sendable {
    var promptCalls = 0
    var processCalls = 0
    var didSampleCalls = 0

    func prompt(_ prompt: MLXArray) {
      self.promptCalls += 1
    }

    func process(logits: MLXArray) -> MLXArray {
      self.processCalls += 1
      return logits
    }

    func didSample(token: MLXArray) {
      self.didSampleCalls += 1
    }
  }
#endif
