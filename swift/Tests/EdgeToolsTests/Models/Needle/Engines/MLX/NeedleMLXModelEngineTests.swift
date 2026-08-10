#if MLX && XGrammar && canImport(MLX) && !os(WASI)
  import EdgeTools
  import Testing
  import CustomDump
  import SnapshotTesting
  import IssueReporting
  import MLX
  import MLXLMCommon

  @Suite(.serialized, .enabledIfMLXTests(), .extendedNeedleInference())
  struct `NeedleMLXModelEngine tests` {
    private typealias Engine = NeedleMLXModelEngine

    private let engine: Engine

    init() async throws {
      self.engine = try await Engine(from: downloadNeedle())
    }

    @Test
    func `Generate Basics`() async throws {
      let generation = try await generateNeedle(using: self.engine)
      expectNoDifference(generation.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.metadata, as: .dump, record: .all)
      }
    }

    @Test
    func `Generate Streamed Response Matches Final Response`() async throws {
      try await expectNeedleStreamedResponseMatchesFinalResponse(using: self.engine)
    }

    @Test
    func `Generate Stops And Returns Stopped Generation`() async throws {
      try await expectNeedleGenerationStops(using: self.engine)
    }

    @Test
    func `Generate Cancels And Throws Cancellation Error`() async throws {
      await expectNeedleGenerationCancellation(using: self.engine)
    }

    @Test
    func `Generate With Untied Word Embeddings`() async throws {
      let directoryURL = try await downloadNeedle()
      let directory = MLXModelDirectory(url: directoryURL)
      var configuration = try directory.loadConfiguration(NeedleModelConfiguration.self)
      configuration.tieWordEmbeddings = false
      let engine = try await Engine(
        from: directory,
        configuration: configuration
      )

      let generation = try await generateNeedle(using: engine)
      expectNoDifference(generation.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.metadata, as: .dump, record: .all)
        assertSnapshot(of: generation.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    func `Generate Through EdgeToolsSession`() async throws {
      let generation = try await generateNeedleThroughSession(using: self.engine)
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
    func `Generate Invokes Custom Logit Processor With Batched Shapes`() async throws {
      let processor = CountingLogitProcessor()

      _ = try await generateNeedle(
        using: self.engine,
        parameters: Engine.GenerateParameters(processor: processor)
      )

      expectNoDifference(processor.promptCalls, 1)
      expectNoDifference(processor.processCalls > 0, true)
      expectNoDifference(processor.didSampleCalls, processor.processCalls)
      expectNoDifference(processor.logitShapes.allSatisfy { $0.count == 2 }, true)
      expectNoDifference(processor.sampledTokenShapes.allSatisfy { $0 == [1] }, true)
    }

    @Test
    func `Generate Records Mean Confidence In Range Zero To One`() async throws {
      let generation = try await generateNeedle(using: self.engine)

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
      await expectNeedleContextLengthExceeded(using: self.engine)
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
    var logitShapes = [[Int]]()
    var sampledTokenShapes = [[Int]]()

    func prompt(_ prompt: MLXArray) {
      self.promptCalls += 1
    }

    func process(logits: MLXArray) -> MLXArray {
      self.processCalls += 1
      self.logitShapes.append(logits.shape)
      return logits
    }

    func didSample(token: MLXArray) {
      self.didSampleCalls += 1
      self.sampledTokenShapes.append(token.shape)
    }
  }
#endif
