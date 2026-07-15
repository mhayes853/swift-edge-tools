import EdgeToolsMacros
import MacroTesting
import SnapshotTesting
import Testing

// MARK: - Suite

@MainActor
@Suite(
  .serialized,
  .macros(
    [
      "EdgeToolsGenerable": EdgeToolsGenerableMacro.self,
      "EdgeToolsIgnored": EdgeToolsIgnoredMacro.self,
      "EdgeToolsGuide": EdgeToolsGuideMacro.self
    ],
    record: .failed
  )
) struct BaseTestSuite {}
