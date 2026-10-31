import EdgeToolsCore
import OrderedCollections

// MARK: - EdgeToolsTemplateFunction

/// A function that an ``EdgeToolsTemplate`` can call or apply as a filter.
///
/// Arguments and results cross into the function as ``EdgeToolsValue`` values. When applied as a
/// filter, the filtered value is the first positional argument.
///
/// ```swift
/// let shout = EdgeToolsTemplateFunction { arguments in
///   guard let text = arguments.positional.first?.string else {
///     throw EdgeToolsTemplateError(code: .renderFailure, message: "shout expects a string")
///   }
///   return .string(text.uppercased() + "!")
/// }
/// let text = try EdgeToolsTemplate(source: "{{ 'hi' | shout }}")
///   .render(functions: ["shout": shout])  // "HI!"
/// ```
public struct EdgeToolsTemplateFunction: Sendable {
  private let body:
    @Sendable (EdgeToolsTemplateArguments) throws(EdgeToolsTemplateError) -> EdgeToolsValue

  /// Creates a function whose failures render with their `String` description.
  public init<Failure: Error>(
    _ body: @escaping @Sendable (EdgeToolsTemplateArguments) throws(Failure) -> EdgeToolsValue
  ) {
    self.init(body) { failure in
      #if $Embedded
        "unknown error"
      #else
        (failure as? any EdgeToolsDescribableError)?.edgeToolsErrorDescription
          ?? String(describing: failure)
      #endif
    }
  }

  /// Creates a function whose failures render with their ``EdgeToolsDescribableError`` description.
  public init<Failure: EdgeToolsDescribableError>(
    _ body: @escaping @Sendable (EdgeToolsTemplateArguments) throws(Failure) -> EdgeToolsValue
  ) {
    self.init(body) { $0.edgeToolsErrorDescription }
  }

  private init<Failure: Error>(
    _ body: @escaping @Sendable (EdgeToolsTemplateArguments) throws(Failure) -> EdgeToolsValue,
    describe: @escaping @Sendable (Failure) -> String
  ) {
    self.body = { arguments throws(EdgeToolsTemplateError) in
      do throws(Failure) {
        return try body(arguments)
      } catch {
        throw EdgeToolsTemplateError(code: .renderFailure, message: describe(error))
      }
    }
  }
}

// MARK: - EdgeToolsTemplateArguments

/// The arguments passed to an ``EdgeToolsTemplateFunction``.
public struct EdgeToolsTemplateArguments: Hashable, Sendable {
  /// The positional arguments in the order they were passed.
  public var positional: [EdgeToolsValue]

  /// The keyword arguments in the order they were passed.
  public var named: OrderedDictionary<String, EdgeToolsValue>

  public init(
    positional: [EdgeToolsValue] = [],
    named: OrderedDictionary<String, EdgeToolsValue> = [:]
  ) {
    self.positional = positional
    self.named = named
  }
}

// MARK: - Template Functions

extension EdgeToolsTemplateFunction {
  var templateFunction: TemplateFunction {
    { _, arguments throws(TemplateFailure) in
      var named = OrderedDictionary<String, EdgeToolsValue>()
      for (name, value) in arguments.named {
        named[name] = value.edgeToolsValue
      }
      let arguments = EdgeToolsTemplateArguments(
        positional: arguments.positional.map(\.edgeToolsValue),
        named: named
      )
      do throws(EdgeToolsTemplateError) {
        return TemplateValue(try self.body(arguments))
      } catch {
        throw TemplateFailure(error.message)
      }
    }
  }
}
