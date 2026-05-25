import MacroTesting
import NeedleMacros
import SnapshotTesting
import Testing

// MARK: - Suite

@MainActor
@Suite(
  .serialized,
  .macros(
    [
      "NeedleGenerable": NeedleGenerableMacro.self,
      "NeedleIgnored": NeedleIgnoredMacro.self,
      "NeedleGuide": NeedleGuideMacro.self
    ],
    record: .failed
  )
) struct BaseTestSuite {}
