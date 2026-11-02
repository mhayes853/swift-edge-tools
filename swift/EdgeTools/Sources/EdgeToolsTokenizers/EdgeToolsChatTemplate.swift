#if ChatTemplates
  import EdgeToolsCore
  import EdgeToolsTemplateRendering
  import OrderedCollections

  // MARK: - EdgeToolsChatTemplateError

  public struct EdgeToolsChatTemplateError: Hashable, Sendable, Error {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let renderFailure = Self(rawValue: "render-failure")
    }

    public let code: Code
    public let message: String

    public init(code: Code, message: String) {
      self.code = code
      self.message = message
    }
  }

  // MARK: - EdgeToolsChatTemplate

  public struct EdgeToolsChatTemplate: Hashable, Sendable {
    public let source: String

    public init(source: String) {
      self.source = source
    }

    /// Renders the template against transformers' chat-template semantics.
    ///
    /// The context may pin `strftime_now` through an `edge_tools_now` key holding whole
    /// seconds since the Unix epoch; it otherwise reads the current time.
    ///
    /// - Parameters:
    ///   - context: The variables the template can read, as a JSON object.
    ///   - cache: The cache to look up the parsed template in.
    /// - Returns: The rendered text.
    public func render(
      context: EdgeToolsValue,
      cache: EdgeToolsTemplateCache = .shared
    ) throws -> String {
      guard case .object(var values) = context else {
        throw EdgeToolsChatTemplateError(
          code: .renderFailure,
          message: "The template context must be a JSON object."
        )
      }
      let now = try renderingInstant(removingFrom: &values)
      do throws(EdgeToolsTemplateError) {
        return try cache.template(source: self.source, options: chatTemplateOptions)
          .render(context: values, functions: [StrftimeNow(instant: now)])
      } catch {
        throw EdgeToolsChatTemplateError(code: .renderFailure, message: error.message)
      }
    }
  }

  // MARK: - Helpers

  private let pinnedNowKey = "edge_tools_now"
  private let chatTemplateOptions = EdgeToolsTemplate.Options(trimBlocks: true, lstripBlocks: true)

  private func renderingInstant(
    removingFrom values: inout OrderedDictionary<String, EdgeToolsValue>
  ) throws(EdgeToolsChatTemplateError) -> Int64 {
    guard let pinned = values.removeValue(forKey: pinnedNowKey) else {
      return currentUnixTime()
    }
    guard case .integer(let seconds) = pinned else {
      throw EdgeToolsChatTemplateError(
        code: .renderFailure,
        message: "`\(pinnedNowKey)` must be whole seconds since the Unix epoch."
      )
    }
    return Int64(seconds)
  }

  private struct StrftimeNow: EdgeToolsTemplateFunction {
    let instant: Int64

    var name: String { "strftime_now" }

    func invoke(
      arguments: EdgeToolsTemplateArguments
    ) throws(EdgeToolsTemplateError) -> EdgeToolsValue {
      guard arguments.positional.count == 1, arguments.named.isEmpty else {
        throw EdgeToolsTemplateError(
          code: .renderFailure,
          message: "\(self.name) must have between 1 and 1 positional arguments and between 0 "
            + "and 0 keyword arguments"
        )
      }
      guard case .string(let format) = arguments.positional[0] else {
        throw EdgeToolsTemplateError(
          code: .renderFailure,
          message: "\(self.name) expects a format string"
        )
      }
      return .string(formattedUTCTime(self.instant, format: format))
    }
  }
#endif
