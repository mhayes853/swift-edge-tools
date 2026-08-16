#if canImport(CMinja)
  import CMinja
  import EdgeToolsCore

  // MARK: - EdgeToolsChatTemplateError

  public struct EdgeToolsChatTemplateError: Hashable, Sendable, Error {
    public struct Code: RawRepresentable, Hashable, Sendable {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public static let invalidArgument = Self(rawValue: "invalid-argument")
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
      let source = Array(self.source.utf8)
      let contextJSON = Array(context.orderedJSONString().utf8)
      var capacity = source.count + contextJSON.count + 4096
      while true {
        var storage = [UInt8](repeating: 0, count: capacity)
        var required = 0
        let status = source.withUnsafeBufferPointer { source in
          contextJSON.withUnsafeBufferPointer { contextJSON in
            storage.withUnsafeMutableBufferPointer { storage in
              edge_template_render(
                source.baseAddress,
                source.count,
                contextJSON.baseAddress,
                contextJSON.count,
                storage.baseAddress,
                storage.count,
                &required
              )
            }
          }
        }
        switch Int(status) {
        case EDGE_TEMPLATE_SUCCESS:
          return String(decoding: storage[..<required], as: UTF8.self)
        case EDGE_TEMPLATE_BUFFER_TOO_SMALL where required > capacity:
          capacity = required
        case EDGE_TEMPLATE_INVALID_ARGUMENT:
          throw EdgeToolsChatTemplateError(code: .invalidArgument, message: lastErrorMessage())
        case EDGE_TEMPLATE_FAILURE:
          throw EdgeToolsChatTemplateError(code: .renderFailure, message: lastErrorMessage())
        default:
          throw EdgeToolsChatTemplateError(
            code: .inconsistentOutput,
            message: "The template renderer reported an inconsistent output size."
          )
        }
      }
    }
  }

  private func lastErrorMessage() -> String {
    edge_template_last_error_message().map { String(cString: $0) }
      ?? "The template renderer encountered an unexpected failure."
  }
#endif
