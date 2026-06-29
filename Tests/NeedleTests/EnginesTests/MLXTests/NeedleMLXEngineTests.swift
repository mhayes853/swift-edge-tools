#if MLX && XGrammar
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
      var tokens = [NeedleToken]()
      let generation = try self.engine.generate(
        prompt: Self.basePrompt,
        parameters: .default,
        onToken: { tokens.append($0) }
      )
      expectNoDifference(generation.wasStopped, false)
      withExpectedIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.metadata, as: .dump, record: .all)
        assertSnapshot(of: generation.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    func `Generate Streamed Response Matches Final Response`() async throws {
      var tokens = [NeedleToken]()
      let generation = try self.engine.generate(
        prompt: Self.basePrompt,
        parameters: .default,
        onToken: { tokens.append($0) }
      )

      let streamedResponse = tokens.map(\.stringValue).joined()
      let finalResponse = generation.tokens.map(\.stringValue).joined()
      expectNoDifference(streamedResponse, finalResponse)
    }

    @Test
    func `Generate Stops And Returns Stopped Generation`() async throws {
      let stopper = self.engine.stopper

      var tokens = [NeedleToken]()
      let generation = try self.engine.generate(
        prompt: Self.basePrompt,
        parameters: .default,
        onToken: {
          tokens.append($0)
          stopper()
        }
      )

      expectNoDifference(generation.wasStopped, true)
      expectNoDifference(tokens.count > 0, true)
      expectNoDifference(generation.decodeMetrics.tokens, tokens.count)
      expectNoDifference(generation.tokens.isEmpty, false)
    }

    @Test
    func `Generate Cancels And Throws Cancellation Error`() async throws {
      let prompt = Self.basePrompt

      // NB: We send and never use the engine outside the task, so this is safe.
      nonisolated(unsafe) let engine = self.engine

      let task = Task {
        _ = try engine.generate(prompt: prompt, parameters: .default) { _ in }
      }

      task.cancel()
      await #expect(throws: CancellationError.self) {
        _ = try await task.value
      }
    }

    @Test
    func `Generate With 4 Bit KV Cache Quantization`() async throws {
      var tokens = [NeedleToken]()
      let generation = try self.engine.generate(
        prompt: Self.basePrompt,
        parameters: NeedleMLXEngine.GenerateParameters(kvCacheQuantizationBits: 4),
        onToken: { tokens.append($0) }
      )
      expectNoDifference(generation.wasStopped, false)
      withExpectedIssue {
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

      var tokens = [NeedleToken]()
      let generation = try engine.generate(
        prompt: Self.basePrompt,
        parameters: .default,
        onToken: { tokens.append($0) }
      )
      expectNoDifference(generation.wasStopped, false)
      withExpectedIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.metadata, as: .dump, record: .all)
        assertSnapshot(of: generation.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    func `Generate Through NeedleSession`() async throws {
      // NB: We send and never use the engine outside the session, so this is safe.
      nonisolated(unsafe) let engine = self.engine
      let session = NeedleSession(engine: engine)
      let generation = try await session.generate(
        tools: [SendEmailTool()],
        with: Self.basePrompt.user
      )
      expectNoDifference(generation.engineGeneration.wasStopped, false)
      withExpectedIssue {
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

      _ = try self.engine.generate(
        prompt: Self.basePrompt,
        parameters: NeedleMLXEngine.GenerateParameters(processor: processor),
        onToken: { _ in }
      )

      expectNoDifference(processor.promptCalls, 1)
      expectNoDifference(processor.processCalls > 0, true)
      expectNoDifference(processor.didSampleCalls, processor.processCalls)
    }

    @Test
    func `Generate Records Mean Confidence In Range Zero To One`() async throws {
      let generation = try self.engine.generate(
        prompt: Self.basePrompt,
        parameters: .default,
        onToken: { _ in }
      )

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

      let error = #expect(throws: NeedleMLXEngineError.self) {
        _ = try self.engine.generate(
          prompt: prompt,
          parameters: .default,
          onToken: { _ in }
        )
      }
      expectNoDifference(error?.message.contains("context length"), true)
    }

    @Test
    func `Tokenize Base`() {
      assertSnapshot(of: self.engine.tokenize(prompt: Self.basePrompt), as: .dump)
    }
  }

  extension `NeedleMLXEngine tests` {
    fileprivate static let basePrompt = NeedlePrompt(
      system: "",
      user: "Send an email to Henry asking him to go on an adventure.",
      tools: [.sendEmail]
    )
  }

  // MARK: - CountingLogitProcessor

  private final class CountingLogitProcessor: LogitProcessor, @unchecked Sendable {
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

  // MARK: - SendEmailTool

  private struct SendEmailTool: NeedleTool {
    @NeedleGenerable
    struct Input: Sendable {
      @NeedleGuide(
        .string(pattern: /[a-z][a-z0-9]{1,10}@gmail\.com/),
        description: "The recipient's email address."
      )
      var address: String

      @NeedleGuide(description: "The subject of an email.")
      var subject: String

      @NeedleGuide(description: "The content of an email.")
      var body: String
    }

    let name = "sendEmail"
    let description = "Sends an email to a recipient with an email address."

    func invoke(input: Input) async throws -> String {
      "Sent email to \(input.address)"
    }
  }
#endif
