import EdgeToolsCore
import OrderedCollections

// MARK: - EdgeToolsTemplate

/// A parsed Jinja template.
///
/// Templates follow the Jinja dialect that tool-calling chat templates use, including Python-style
/// string methods, filters, macros, call blocks, loop controls, and namespaces. Parsing happens
/// once when the template is created, so a template can be rendered any number of times, from any
/// thread, against different contexts.
///
/// ```swift
/// let template = try EdgeToolsTemplate(source: "Hello {{ name | upper }}!")
/// let text = try template.render(context: ["name": "edge tools"])  // "Hello EDGE TOOLS!"
/// ```
public struct EdgeToolsTemplate: Sendable {
  /// Whitespace handling applied when parsing a template.
  public struct Options: Hashable, Sendable {
    /// Removes the first newline after a block tag.
    public var trimBlocks: Bool

    /// Removes the spaces and tabs that precede a tag on its line.
    public var lstripBlocks: Bool

    /// Keeps a single trailing newline at the end of the template.
    public var keepTrailingNewline: Bool

    public init(
      trimBlocks: Bool = false,
      lstripBlocks: Bool = false,
      keepTrailingNewline: Bool = false
    ) {
      self.trimBlocks = trimBlocks
      self.lstripBlocks = lstripBlocks
      self.keepTrailingNewline = keepTrailingNewline
    }
  }

  /// The source the template was parsed from.
  public let source: String

  /// The options the template was parsed with.
  public let options: Options

  /// An approximate number of bytes this template holds in memory, including its source.
  ///
  /// The estimate scales with the size of the source and the number of syntax nodes it parses
  /// into. It's meant for budgeting, such as the memory limit of an ``EdgeToolsTemplateCache``,
  /// so expect it to differ from the true allocation size by up to about 20%, and by more for
  /// templates under 1 KB.
  public let estimatedMemorySize: Int

  private let parsedSource: [UInt8]
  private let root: TemplateNode

  /// Parses a template.
  ///
  /// - Parameters:
  ///   - source: The template source.
  ///   - options: The whitespace handling to parse with.
  /// - Throws: An ``EdgeToolsTemplateError`` with the ``EdgeToolsTemplateError/Code/invalidSyntax``
  ///   code when the source is not a valid template.
  public init(source: String, options: Options = Options()) throws(EdgeToolsTemplateError) {
    self.source = source
    self.options = options
    self.parsedSource = normalizedNewlines(Array(source.utf8))
    do throws(TemplateFailure) {
      self.root = try TemplateParser.parse(self.parsedSource, options: options)
    } catch {
      throw EdgeToolsTemplateError(
        code: .invalidSyntax,
        message: error.message(in: self.parsedSource)
      )
    }
    self.estimatedMemorySize = templateMemorySize(
      sourceSize: self.parsedSource.count,
      root: self.root
    )
  }

  /// Renders the template.
  ///
  /// - Parameters:
  ///   - context: The variables the template can read.
  ///   - functions: Functions the template can call or apply as filters by their name, which
  ///     take precedence over variables of the same name. When several functions share a name,
  ///     the last one wins.
  /// - Returns: The rendered text.
  /// - Throws: An ``EdgeToolsTemplateError`` with the ``EdgeToolsTemplateError/Code/renderFailure``
  ///   code when rendering fails.
  public func render(
    context: OrderedDictionary<String, EdgeToolsValue> = [:],
    functions: [any EdgeToolsTemplateFunction] = []
  ) throws(EdgeToolsTemplateError) -> String {
    let scope = TemplateContext(parent: .builtins())
    for (name, value) in context {
      scope.set(name, TemplateValue(value))
    }
    for function in functions {
      scope.set(function.name, .function(templateFunction(function)))
    }
    do throws(TemplateFailure) {
      return try self.root.render(scope)
    } catch {
      throw EdgeToolsTemplateError(
        code: .renderFailure,
        message: error.message(in: self.parsedSource)
      )
    }
  }
}

// MARK: - Helpers

private func normalizedNewlines(_ source: [UInt8]) -> [UInt8] {
  #if os(Windows)
    source.indices.compactMap { index in
      let isCarriageReturn =
        source[index] == UInt8(ascii: "\r") && index + 1 < source.count
        && source[index + 1] == .newline
      return isCarriageReturn ? nil : source[index]
    }
  #else
    source
  #endif
}
