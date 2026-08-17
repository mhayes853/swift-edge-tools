#if ChatTemplates && canImport(CMinja)
  import CMinja
  import EdgeToolsCore

  // MARK: - EdgeToolsChatTemplateError

  public struct EdgeToolsChatTemplateError: Hashable, Sendable, Error {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let renderFailure = Self(rawValue: "render-failure")
      public static let inconsistentOutput = Self(rawValue: "inconsistent-output")
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
      let contextJSON = context.orderedJSONString()
      let capacity = self.source.utf8.count + contextJSON.utf8.count + 4096
      return try self.source.withCString { source in
        try contextJSON.withCString { contextJSON in
          var storage = [CChar](repeating: 0, count: capacity)
          let required = storage.withUnsafeMutableBufferPointer {
            edge_template_render(source, contextJSON, $0.baseAddress, $0.count)
          }
          guard required > 0 else {
            throw EdgeToolsChatTemplateError(code: .renderFailure, message: lastErrorMessage())
          }
          guard required > capacity else {
            return String(cString: storage)
          }
          storage = [CChar](repeating: 0, count: required)
          let refilled = storage.withUnsafeMutableBufferPointer {
            edge_template_render(source, contextJSON, $0.baseAddress, $0.count)
          }
          guard refilled == required else {
            throw EdgeToolsChatTemplateError(
              code: .inconsistentOutput,
              message: "The template renderer reported an inconsistent output size."
            )
          }
          return String(cString: storage)
        }
      }
    }
  }

  private func lastErrorMessage() -> String {
    edge_template_last_error_message().map { String(cString: $0) }
      ?? "The template renderer encountered an unexpected failure."
  }
#endif
