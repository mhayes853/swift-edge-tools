#if CoreML && canImport(CoreML)
  import CoreML

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  extension NeedleConfidenceState {
    mutating func add(logits: MLTensor) {

    }
  }
#endif
