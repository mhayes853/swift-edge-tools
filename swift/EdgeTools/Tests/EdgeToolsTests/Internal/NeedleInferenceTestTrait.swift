import Foundation
import Testing

extension Trait where Self == ConditionTrait {
  static func basicNeedleInference(
    sourceLocation: SourceLocation = #_sourceLocation
  ) -> Self {
    .enabled(
      if: needleInferenceTestMode == "basic",
      "Runs only in the basic Needle inference configuration.",
      sourceLocation: sourceLocation
    )
  }

  static func extendedNeedleInference(
    sourceLocation: SourceLocation = #_sourceLocation
  ) -> Self {
    .disabled(
      if: needleInferenceTestMode == "basic",
      "Excluded from the basic Needle inference configuration.",
      sourceLocation: sourceLocation
    )
  }
}

private var needleInferenceTestMode: String? {
  ProcessInfo.processInfo.environment["NEEDLE_INFERENCE_TESTS"]?
    .trimmingCharacters(in: .whitespacesAndNewlines)
    .lowercased()
}
