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
      let metrics = try engine.prefill(prompt: .base)
      withExpectedIssue { assertSnapshot(of: metrics, as: .dump, record: .all) }
    }

    @Test
    func `Prefill Reduces Prefilled Tokens When Generating`() async throws {
      let prefillMetrics = try engine.prefill(prompt: .base)

      let matcher = try await self.engine.grammarEngine.compile(tools: [.sendEmail])
      let generation = try engine.generate(
        prompt: NeedlePrompt(prefillable: .base, tools: [.sendEmail]),
        matcher: matcher,
        onToken: { print($0) }
      )
      expectNoDifference(generation.prefillMetrics.tokens < prefillMetrics.tokens, true)
    }

    @Test
    func `Ignores Prefill Tokens When Generating With Different Prompt`() async throws {
      let prefillMetrics = try self.engine.prefill(prompt: .base)
      let matcher = try await self.engine.grammarEngine.compile(tools: [.sendEmail])
      let generation = try self.engine.generate(
        prompt: NeedlePrompt(
          system: "You are a helpful assistant.",
          user: "Write an email to Henry telling him to meet the emperor.",
          tools: [.sendEmail]
        ),
        matcher: matcher,
        onToken: { _ in }
      )
      expectNoDifference(generation.prefillMetrics.tokens > prefillMetrics.tokens, true)
    }

    @Test
    func `Ignores Prefill Tokens After Reset`() async throws {
      let prefillMetrics = try self.engine.prefill(prompt: .base)
      let matcher = try await self.engine.grammarEngine.compile(tools: [.sendEmail])

      self.engine.reset()

      let generation = try self.engine.generate(
        prompt: NeedlePrompt(prefillable: .base, tools: [.sendEmail]),
        matcher: matcher,
        onToken: { _ in }
      )
      expectNoDifference(generation.prefillMetrics.tokens >= prefillMetrics.tokens, true)
    }

    @Test
    func `Generate Basics`() async throws {
      let matcher = try await self.engine.grammarEngine.compile(tools: [.sendEmail])

      var tokens = [NeedleToken]()
      let generation = try engine.generate(
        prompt: NeedlePrompt(prefillable: .base, tools: [.sendEmail]),
        matcher: matcher,
        onToken: { tokens.append($0) }
      )
      withExpectedIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: tokens, as: .dump, record: .all)
      }
    }
  }

  extension NeedlePrefillablePrompt {
    fileprivate static let base = Self(
      system: "You are a helpful assistant who can send emails.",
      user: "Send an email to Henry about his wonderful adventures."
    )
  }
#endif
