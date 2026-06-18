#if SwiftNeedleMLX && SwiftNeedleXGrammar
  import Needle
  import Testing
  import CustomDump
  import SnapshotTesting
  import IssueReporting
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
        prompt: NeedlePrompt(prefillable: .base, tools: [.sendEmail]),
        matcher: matcher,
        onToken: { tokens.append($0) }
      )
      expectNoDifference(generation.wasStoped, false)
      withExpectedIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: tokens, as: .dump, record: .all)
      }
    }

    @Test
    func `Generate Stops And Returns Stopped Generation`() async throws {
      let matcher = try await self.engine.grammarEngine.compile(tools: [.sendEmail])
      let stopper = self.engine.stopper

      var tokens = [NeedleToken]()
      let generation = try self.engine.generate(
        prompt: NeedlePrompt(prefillable: .base, tools: [.sendEmail]),
        matcher: matcher,
        onToken: {
          tokens.append($0)
          stopper()
        }
      )

      expectNoDifference(generation.wasStoped, true)
      expectNoDifference(tokens.count > 0, true)
      expectNoDifference(generation.decodeMetrics.tokens, tokens.count)
      expectNoDifference(generation.response.isEmpty, false)
    }

    @Test
    func `Generate Cancels And Throws Cancellation Error`() async throws {
      let prompt = NeedlePrompt(prefillable: .base, tools: [.sendEmail])

      // NB: We send and never use the engine/matcher outside the task, so this is safe.
      nonisolated(unsafe) let engine = self.engine
      nonisolated(unsafe) let matcher = try await engine.grammarEngine.compile(tools: [.sendEmail])

      let task = Task {
        _ = try engine.generate(prompt: prompt, matcher: matcher) { _ in }
      }

      task.cancel()
      await #expect(throws: CancellationError.self) {
        _ = try await task.value
      }
    }
  }

  extension NeedlePrefillablePrompt {
    fileprivate static let base = Self(
      system: "You are a helpful assistant who must send emails.",
      user: "Send an email to Henry about his adventures."
    )
  }
#endif
