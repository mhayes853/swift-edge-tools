#if MLX && XGrammar && canImport(MLX)
  import EdgeTools
  import Testing
  import CustomDump
  import SnapshotTesting
  import IssueReporting
  import MLX
  import MLXLMCommon

  @Suite(.serialized, .enabledIfXcode())
  struct `NeedleMLXEngine tests` {
    private typealias Engine = EdgeToolsMLXEngine<NeedleMLXModel>

    private let engine: Engine

    init() async throws {
      self.engine = try await Engine(from: downloadNeedle())
    }

    @Test
    func `Generate Basics`() async throws {
      let tokens = Lock([EdgeToolsToken]())
      let generationTask = try self.engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel(
          onToken: { token in tokens.withLock { $0.append(token) } }
        )
      )
      let generation = try await generationTask.value
      expectNoDifference(generation.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.metadata, as: .dump, record: .all)
      }
    }

    @Test
    func `Generate Streamed Response Matches Final Response`() async throws {
      let tokens = Lock([EdgeToolsToken]())
      let generationTask = try self.engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel(
          onToken: { token in tokens.withLock { $0.append(token) } }
        )
      )
      let generation = try await generationTask.value

      let streamedResponse = tokens.withLock { $0.map(\.stringValue).joined() }
      let finalResponse = generation.response
      expectNoDifference(streamedResponse, finalResponse)
    }

    @Test
    func `Generate Stops And Returns Stopped Generation`() async throws {
      let tokens = Lock([EdgeToolsToken]())
      let generationTaskBox = Lock<Engine.GenerationTask?>(nil)
      let generationTask = try self.engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel(
          onToken: { token in
            tokens.withLock { $0.append(token) }
            generationTaskBox.withLock { $0?.stop() }
          }
        )
      )
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
          tools: NeedlePrompt.sendAdventureEmailDefinitions,
          parameters: .default,
          channel: EdgeToolsGenerationChannel()
        )
        _ = try await generationTask.value
      }

      task.cancel()
      await #expect(throws: CancellationError.self) {
        _ = try await task.value
      }
    }

    @Test
    func `Generate With 4 Bit KV Cache Quantization`() async throws {
      let tokenStorage = Lock([EdgeToolsToken]())
      let generationTask = try self.engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: Engine.GenerateParameters(kvCacheQuantizationBits: 4),
        channel: EdgeToolsGenerationChannel(
          onToken: { token in tokenStorage.withLock { $0.append(token) } }
        )
      )
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
      let engine = try await Engine(
        from: downloadNeedle(),
        model: { configuration in
          var configuration = configuration
          configuration.tieWordEmbeddings = false
          return NeedleMLXModel(configuration: configuration)
        }
      )

      let tokenStorage = Lock([EdgeToolsToken]())
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel(
          onToken: { token in tokenStorage.withLock { $0.append(token) } }
        )
      )
      let generation = try await generationTask.value
      expectNoDifference(generation.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.metadata, as: .dump, record: .all)
        assertSnapshot(of: generation.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    func `Generate Through EdgeToolsSession`() async throws {
      let session = EdgeToolsSession(engine: self.engine)
      let generation = try await session.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailTools
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
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: Engine.GenerateParameters(processor: processor),
        channel: EdgeToolsGenerationChannel()
      )
      _ = try await generationTask.value

      expectNoDifference(processor.promptCalls, 1)
      expectNoDifference(processor.processCalls > 0, true)
      expectNoDifference(processor.didSampleCalls, processor.processCalls)
    }

    @Test
    func `Generate Records Mean Confidence In Range Zero To One`() async throws {
      let generationTask = try self.engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let generation = try await generationTask.value

      let confidence = try #require(generation.metadata.generationConfidence)
      expectNoDifference((0...1).contains(confidence), true)

      let perToken = try #require(generation.metadata.perTokenConfidences)
      expectNoDifference(perToken.count, generation.tokens.count)
      for uncertainty in perToken {
        expectNoDifference((0...1).contains(uncertainty), true)
      }
    }

    @Test
    func `Generate Throws When Prompt Exceeds Context Length`() async throws {
      let prompt = NeedlePrompt(
        system: "",
        user: String(repeating: "token ", count: 2_000)
      )

      let error = await #expect(throws: EdgeToolsError.self) {
        let generationTask = try self.engine.generate(
          prompt: prompt,
          parameters: .default,
          channel: EdgeToolsGenerationChannel()
        )
        _ = try await generationTask.value
      }
      expectNoDifference(error?.code, .contextLengthExceeded)
    }

    @Test
    func `Tokenize Base`() async throws {
      let tokens = try await self.engine.tokenize(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions
      )
      assertSnapshot(of: tokens, as: .dump)
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
