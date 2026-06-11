#if SwiftNeedleMLX
  import Needle
  import Testing
  import CustomDump
  import SnapshotTesting
  import IssueReporting

  @Suite(.serialized, .enabledIfXcode())
  struct `NeedleMLXEnegine tests` {
    private let engine: NeedleMLXEngine<AlwaysGrammarEngine>

    init() throws {
      fatalError()
    }

    @Test
    func `Prefill Basics`() throws {
      let metrics = try engine.prefill(prompt: .prefill)
      withExpectedIssue { assertSnapshot(of: metrics, as: .json) }
    }

    @Test
    func `Prefill Reduces Prefilled Tokens When Generating`() throws {
      let prefillMetrics = try engine.prefill(prompt: .prefill)

      var prompt = NeedlePrompt.prefill
      prompt.user = "Send an email to Henry."
      let generation = try engine.generate(
        prompt: prompt,
        matcher: AlwaysGrammarEngine.Matcher(),
        onToken: { _ in }
      )
      expectNoDifference(generation.prefillMetrics.tokens < prefillMetrics.tokens, true)
    }

    @Test
    func `Ignores Prefill Tokens When Generating With Different Prompt`() throws {
      let prefillMetrics = try engine.prefill(prompt: .prefill)
      let generation = try engine.generate(
        prompt: NeedlePrompt(
          system: "You are a helpful assistant.",
          user: "Write an email to Henry, and destroy the galaxy.",
          tools: [.sendEmail]
        ),
        matcher: AlwaysGrammarEngine.Matcher(),
        onToken: { _ in }
      )
      expectNoDifference(generation.prefillMetrics.tokens > prefillMetrics.tokens, true)
    }

    @Test
    func `Generate Basics`() throws {
      var prompt = NeedlePrompt.prefill
      prompt.user = "Send an email to Henry"

      let generation = try engine.generate(
        prompt: prompt,
        matcher: AlwaysGrammarEngine.Matcher(),
        onToken: { _ in }
      )
      withExpectedIssue { assertSnapshot(of: generation, as: .json) }
    }

    @Test
    func `Constrained Generation Basics`() throws {
      // TODO
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
