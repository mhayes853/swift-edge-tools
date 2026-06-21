#if SwiftNeedleMLX && SwiftNeedleXGrammar
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
      let matcher = try await self.engine.grammarEngine.compile(tools: [.sendEmail])

      var tokens = [NeedleToken]()
      let generation = try self.engine.generate(
        prompt: Self.basePrompt,
        matcher: matcher,
        parameters: .default,
        onToken: { tokens.append($0) }
      )
      expectNoDifference(generation.wasStoped, false)
      withExpectedIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.tokens.map(\.stringValue).joined(), as: .dump, record: .all)
      }
    }

    @Test
    func `Generate Streamed Response Matches Final Response`() async throws {
      let matcher = try await self.engine.grammarEngine.compile(tools: [.sendEmail])

      var tokens = [NeedleToken]()
      let generation = try self.engine.generate(
        prompt: Self.basePrompt,
        matcher: matcher,
        parameters: .default,
        onToken: { tokens.append($0) }
      )

      let streamedResponse = tokens.map(\.stringValue).joined()
      let finalResponse = generation.tokens.map(\.stringValue).joined()
      expectNoDifference(streamedResponse, finalResponse)
    }

    @Test
    func `Generate Stops And Returns Stopped Generation`() async throws {
      let matcher = try await self.engine.grammarEngine.compile(tools: [.sendEmail])
      let stopper = self.engine.stopper

      var tokens = [NeedleToken]()
      let generation = try self.engine.generate(
        prompt: Self.basePrompt,
        matcher: matcher,
        parameters: .default,
        onToken: {
          tokens.append($0)
          stopper()
        }
      )

      expectNoDifference(generation.wasStoped, true)
      expectNoDifference(tokens.count > 0, true)
      expectNoDifference(generation.decodeMetrics.tokens, tokens.count)
      expectNoDifference(generation.tokens.isEmpty, false)
    }

    @Test
    func `Generate Cancels And Throws Cancellation Error`() async throws {
      let prompt = Self.basePrompt

      // NB: We send and never use the engine/matcher outside the task, so this is safe.
      nonisolated(unsafe) let engine = self.engine
      nonisolated(unsafe) let matcher = try await engine.grammarEngine.compile(tools: [.sendEmail])

      let task = Task {
        _ = try engine.generate(
          prompt: prompt,
          matcher: matcher,
          parameters: .default,
        ) { _ in }
      }

      task.cancel()
      await #expect(throws: CancellationError.self) {
        _ = try await task.value
      }
    }

    @Test
    func `Generate With 4 Bit KV Cache Quantization`() async throws {
      let matcher = try await self.engine.grammarEngine.compile(tools: [.sendEmail])

      var tokens = [NeedleToken]()
      let generation = try self.engine.generate(
        prompt: Self.basePrompt,
        matcher: matcher,
        parameters: NeedleMLXEngine.GenerateParamaters(kvCacheQuantizationBits: 4),
        onToken: { tokens.append($0) }
      )
      expectNoDifference(generation.wasStoped, false)
      withExpectedIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.tokens.map(\.stringValue).joined(), as: .dump, record: .all)
      }
    }

    @Test
    func `Generate Invokes Custom Logit Processor`() async throws {
      let matcher = try await self.engine.grammarEngine.compile(tools: [.sendEmail])
      let processor = CountingLogitProcessor()

      _ = try self.engine.generate(
        prompt: Self.basePrompt,
        matcher: matcher,
        parameters: NeedleMLXEngine.GenerateParamaters(processor: processor),
        onToken: { _ in }
      )

      expectNoDifference(processor.promptCalls, 1)
      expectNoDifference(processor.processCalls > 0, true)
      expectNoDifference(processor.didSampleCalls, processor.processCalls)
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
#endif
