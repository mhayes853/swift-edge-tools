import Foundation
import Testing

extension Trait where Self == ConditionTrait {
  static func multimodal(sourceLocation: SourceLocation = #_sourceLocation) -> Self {
    .disabled(if: isMultimodalTestsDisabled(), sourceLocation: sourceLocation)
  }
}

private func isMultimodalTestsDisabled() -> Bool {
  guard let value = ProcessInfo.processInfo.environment["EDGE_TOOLS_DISABLE_MULTIMODAL_TESTS"] else {
    return false
  }
  let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
  guard !normalized.isEmpty else { return false }
  return normalized != "0" && normalized != "false"
}
