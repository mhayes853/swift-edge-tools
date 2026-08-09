import Foundation
import Testing

func isMLXTestsEnabled() -> Bool {
  ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
    || ProcessInfo.processInfo.environment["EDGE_TOOLS_ENABLE_MLX_TESTS"] == "1"
}

extension Trait where Self == ConditionTrait {
  static func enabledIfMLXTests(sourceLocation: SourceLocation = #_sourceLocation) -> Self {
    .enabled(if: isMLXTestsEnabled(), sourceLocation: sourceLocation)
  }
}
