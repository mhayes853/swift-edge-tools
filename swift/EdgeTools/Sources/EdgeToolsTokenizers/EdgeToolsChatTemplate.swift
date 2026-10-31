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
    public func render(context: EdgeToolsValue) throws -> String {
      guard case .object(var values) = context else {
        throw EdgeToolsChatTemplateError(
          code: .renderFailure,
          message: "The template context must be a JSON object."
        )
      }
      let now = try renderingInstant(removingFrom: &values)
      do throws(EdgeToolsTemplateError) {
        return try cachedTemplate(source: self.source)
          .render(context: values, functions: ["strftime_now": strftimeNow(at: now)])
      } catch {
        throw EdgeToolsChatTemplateError(code: .renderFailure, message: error.message)
      }
    }
  }

  // MARK: - Template Cache

  private let maxCachedTemplates = 4
  private let cachedTemplates = Lock([EdgeToolsTemplate]())

  private func cachedTemplate(source: String) throws(EdgeToolsTemplateError) -> EdgeToolsTemplate {
    let cached = cachedTemplates.withLock { templates -> EdgeToolsTemplate? in
      guard let index = templates.firstIndex(where: { $0.source == source }) else {
        return nil
      }
      let template = templates.remove(at: index)
      templates.insert(template, at: 0)
      return template
    }
    if let cached {
      return cached
    }
    let template = try EdgeToolsTemplate(
      source: source,
      options: EdgeToolsTemplate.Options(trimBlocks: true, lstripBlocks: true)
    )
    cachedTemplates.withLock { templates in
      templates.removeAll { $0.source == source }
      templates.insert(template, at: 0)
      if templates.count > maxCachedTemplates {
        templates.removeLast()
      }
    }
    return template
  }

  // MARK: - Helpers

  private let pinnedNowKey = "edge_tools_now"

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

  private func strftimeNow(at instant: Int64) -> EdgeToolsTemplateFunction {
    EdgeToolsTemplateFunction { arguments throws(EdgeToolsTemplateError) in
      guard arguments.positional.count == 1, arguments.named.isEmpty else {
        throw EdgeToolsTemplateError(
          code: .renderFailure,
          message: "strftime_now must have between 1 and 1 positional arguments and between 0 "
            + "and 0 keyword arguments"
        )
      }
      guard case .string(let format) = arguments.positional[0] else {
        throw EdgeToolsTemplateError(
          code: .renderFailure,
          message: "strftime_now expects a format string"
        )
      }
      return .string(formattedUTCTime(instant, format: format))
    }
  }
#endif
