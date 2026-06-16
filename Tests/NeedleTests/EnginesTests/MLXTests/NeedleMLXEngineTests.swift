#if SwiftNeedleMLX && SwiftNeedleXGrammar
  import Needle
  import Testing
  import CustomDump
  import SnapshotTesting
  import IssueReporting

  @Suite(.serialized, .enabledIfXcode())
  struct `NeedleMLXEnegine tests` {
    private let engine: NeedleMLXEngine

    init() async throws {
      self.engine = try await NeedleMLXEngine(from: downloadNeedleHF())
    }

    @Test
    func `Prefill Basics`() throws {
      let metrics = try engine.prefill(prompt: .prefill)
      withExpectedIssue { assertSnapshot(of: metrics, as: .dump, record: .all) }
    }

    @Test
    func `Prefill Reduces Prefilled Tokens When Generating`() async throws {
      let prefillMetrics = try engine.prefill(prompt: .prefill)

      let matcher = try await self.engine.grammarEngine.compile(tools: [.sendEmail])

      var prompt = NeedlePrompt.prefill
      prompt.user = "Send an email to Henry."
      let generation = try engine.generate(
        prompt: prompt,
        matcher: matcher,
        onToken: { _ in }
      )
      expectNoDifference(generation.prefillMetrics.tokens < prefillMetrics.tokens, true)
    }

    @Test
    func `Ignores Prefill Tokens When Generating With Different Prompt`() async throws {
      let prefillMetrics = try self.engine.prefill(prompt: .prefill)
      let matcher = try await self.engine.grammarEngine.compile(tools: [.sendEmail])
      let generation = try self.engine.generate(
        prompt: NeedlePrompt(
          system: "You are a helpful assistant.",
          user: "Write an email to Henry, and destroy the galaxy.",
          tools: [.sendEmail]
        ),
        matcher: matcher,
        onToken: { _ in }
      )
      expectNoDifference(generation.prefillMetrics.tokens > prefillMetrics.tokens, true)
    }

    @Test
    func `Generate Basics`() async throws {
      var prompt = NeedlePrompt.prefill
      prompt.user = "Send an email to Henry"

      let matcher = try await self.engine.grammarEngine.compile(tools: [.sendEmail])

      var tokens = [NeedleToken]()
      let generation = try engine.generate(
        prompt: prompt,
        matcher: matcher,
        onToken: { tokens.append($0) }
      )
      withExpectedIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: tokens, as: .dump, record: .all)
      }
    }
  }

  extension NeedlePrompt {
    fileprivate static let prefill = Self(
      system: "",
      user: "Send an email to ",
      tools: [.sendEmail]
    )
  }
#endif
