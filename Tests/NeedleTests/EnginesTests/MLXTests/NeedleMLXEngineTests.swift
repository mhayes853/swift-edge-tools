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
