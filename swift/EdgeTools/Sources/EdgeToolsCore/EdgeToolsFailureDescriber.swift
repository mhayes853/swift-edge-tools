// MARK: - EdgeToolsFailureDescriber

/// A type that describes its failures in text.
///
/// Failures that conform to ``EdgeToolsDescribableError`` are described with their
/// ``EdgeToolsDescribableError/edgeToolsErrorDescription`` by default. Other failures are described
/// with `String(describing:)`, or as "unknown error" in Embedded Swift. A conforming type that is
/// generic over its failure always uses the latter default, so implement ``describeFailure(_:)``
/// directly in that case.
public protocol EdgeToolsFailureDescriber {
  /// The error that this type throws.
  associatedtype Failure: Error

  /// Describes a failure thrown by this type.
  ///
  /// - Parameter failure: The thrown failure.
  /// - Returns: A description of the failure.
  func describeFailure(_ failure: Failure) -> String
}

extension EdgeToolsFailureDescriber {
  public func describeFailure(_ failure: Failure) -> String {
    #if $Embedded
      "unknown error"
    #else
      String(describing: failure)
    #endif
  }
}

extension EdgeToolsFailureDescriber where Failure: EdgeToolsDescribableError {
  public func describeFailure(_ failure: Failure) -> String {
    failure.edgeToolsErrorDescription
  }
}
