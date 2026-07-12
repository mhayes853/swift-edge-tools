import Foundation
import Testing

func isRunningTestsFromXcode() -> Bool {
  ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
}

extension Trait where Self == ConditionTrait {
  static func enabledIfXcode(sourceLocation: SourceLocation = #_sourceLocation) -> Self {
    .enabled(if: isRunningTestsFromXcode(), sourceLocation: sourceLocation)
  }
}
