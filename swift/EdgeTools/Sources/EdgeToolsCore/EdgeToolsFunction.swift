// MARK: - EdgeToolsFunction

/// A named function whose failures can be described in text.
///
/// Models call tools and templates call template functions by their ``name``. When a call fails,
/// ``describeFailure(_:)`` turns the failure into the message that is reported back.
///
/// Failures that conform to ``EdgeToolsDescribableError`` are described with their
/// ``EdgeToolsDescribableError/edgeToolsErrorDescription`` by default. Other failures are described
/// with `String(describing:)`, or as "unknown error" in Embedded Swift. A conforming type that is
/// generic over its failure always uses the latter default, so implement ``describeFailure(_:)``
/// directly in that case.
public protocol EdgeToolsFunction: Sendable {
  /// The error that calling this function throws.
  associatedtype Failure: Error

  /// The name that callers refer to this function by.
  var name: String { get }

  /// Describes a failure thrown by this function.
  ///
  /// - Parameter failure: The thrown failure.
  /// - Returns: A description of the failure.
  func describeFailure(_ failure: Failure) -> String
}

extension EdgeToolsFunction {
  public func describeFailure(_ failure: Failure) -> String {
    #if $Embedded
      "unknown error"
    #else
      String(describing: failure)
    #endif
  }
}

extension EdgeToolsFunction where Failure: EdgeToolsDescribableError {
  public func describeFailure(_ failure: Failure) -> String {
    failure.edgeToolsErrorDescription
  }
}
