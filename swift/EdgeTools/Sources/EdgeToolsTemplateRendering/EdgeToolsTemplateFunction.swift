import EdgeToolsCore
import OrderedCollections

// MARK: - EdgeToolsTemplateFunction

/// A function that an ``EdgeToolsTemplate`` can call or apply as a filter.
///
/// Arguments and results cross into the function as ``EdgeToolsValue`` values. When applied as a
/// filter, the filtered value is the first positional argument.
///
/// ```swift
/// struct Shout: EdgeToolsTemplateFunction {
///   var name: String { "shout" }
///
///   func invoke(
///     arguments: EdgeToolsTemplateArguments
///   ) throws(EdgeToolsTemplateError) -> EdgeToolsValue {
///     guard let text = arguments.positional.first?.string else {
///       throw EdgeToolsTemplateError(code: .renderFailure, message: "shout expects a string")
///     }
///     return .string(text.uppercased() + "!")
///   }
/// }
///
/// let text = try EdgeToolsTemplate(source: "{{ 'hi' | shout }}")
///   .render(functions: [Shout()])  // "HI!"
/// ```
public protocol EdgeToolsTemplateFunction: EdgeToolsFunctionDescriptor {
  associatedtype Failure = EdgeToolsTemplateError

  /// Invokes this function.
  ///
  /// - Parameter arguments: The arguments the template passed.
  /// - Returns: The value the call evaluates to.
  func invoke(arguments: EdgeToolsTemplateArguments) throws(Failure) -> EdgeToolsValue

  /// Invokes this function, converting its failures into an ``EdgeToolsTemplateError``.
  ///
  /// The default implementation calls ``invoke(arguments:)`` and describes failures with
  /// ``EdgeToolsFunctionDescriptor/describeFailure(_:)``.
  ///
  /// - Parameter arguments: The arguments the template passed.
  /// - Returns: The value the call evaluates to.
  func templateCall(
    arguments: EdgeToolsTemplateArguments
  ) throws(EdgeToolsTemplateError) -> EdgeToolsValue
}

extension EdgeToolsTemplateFunction {
  public func templateCall(
    arguments: EdgeToolsTemplateArguments
  ) throws(EdgeToolsTemplateError) -> EdgeToolsValue {
    do throws(Failure) {
      return try self.invoke(arguments: arguments)
    } catch {
      throw EdgeToolsTemplateError(code: .renderFailure, message: self.describeFailure(error))
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

func templateFunction(_ function: any EdgeToolsTemplateFunction) -> TemplateFunction {
  { _, arguments throws(TemplateFailure) in
    var named = OrderedDictionary<String, EdgeToolsValue>()
    for (name, value) in arguments.named {
      named[name] = value.edgeToolsValue
    }
    let arguments = EdgeToolsTemplateArguments(
      positional: arguments.positional.map { $0.edgeToolsValue },
      named: named
    )
    do throws(EdgeToolsTemplateError) {
      return TemplateValue(try function.templateCall(arguments: arguments))
    } catch {
      throw TemplateFailure(error.message)
    }
  }
}
