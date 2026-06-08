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
      let metrics = try engine.prefill(prompt: "Send an email to ", tools: [.sendEmail])
      withExpectedIssue { assertSnapshot(of: metrics, as: .json) }
    }

    @Test
    func `Prefill Reduces Prefilled Tokens When Generating`() throws {
      let prefillMetrics = try engine.prefill(prompt: "Send an email to ", tools: [.sendEmail])
      let generation = try engine.generate(
        prompt: "Send an email to Henry.",
        tools: [.sendEmail],
        matcher: AlwaysGrammarEngine.Matcher(),
        onToken: { _ in }
      )
      expectNoDifference(generation.prefillMetrics.tokens < prefillMetrics.tokens, true)
    }

    @Test
    func `Ignores Prefill Tokens When Generating With Different Prompt`() throws {
      let prefillMetrics = try engine.prefill(prompt: "Send an email to ", tools: [.sendEmail])
      let generation = try engine.generate(
        prompt: "Write an email to Henry, and destroy the galaxy.",
        tools: [.sendEmail],
        matcher: AlwaysGrammarEngine.Matcher(),
        onToken: { _ in }
      )
      expectNoDifference(generation.prefillMetrics.tokens > prefillMetrics.tokens, true)
    }

    @Test
    func `Generate Basics`() throws {
      let generation = try engine.generate(
        prompt: "Send an email to Henry.",
        tools: [.sendEmail],
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
#endif
