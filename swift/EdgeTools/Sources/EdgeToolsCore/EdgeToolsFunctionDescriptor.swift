// MARK: - EdgeToolsFunctionDescriptor

/// A named function whose failures can be described in text.
///
/// Models call tools and templates call template functions by their ``name``. When a call fails,
/// ``EdgeToolsFailureDescriber/describeFailure(_:)`` turns the failure into the message that is
/// reported back.
public protocol EdgeToolsFunctionDescriptor: EdgeToolsFailureDescriber, Sendable {
  /// The name that callers refer to this function by.
  var name: String { get }
}
