import Foundation
import Testing

extension Trait where Self == ConditionTrait {
  static func experimental(sourceLocation: SourceLocation = #_sourceLocation) -> Self {
    .disabled(if: isExperimentalTestsDisabled(), sourceLocation: sourceLocation)
  }
}

private func isExperimentalTestsDisabled() -> Bool {
  guard let value = ProcessInfo.processInfo.environment["EXPERIMENTAL_TESTS"] else {
    return false
  }
  let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
  guard !normalized.isEmpty else { return false }
  return normalized != "0" && normalized != "false"
}
