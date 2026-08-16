import EdgeToolsCore

#if FoundationEssentials
  import _EdgeToolsFoundation
#endif

// MARK: - EdgeToolsAutoTokenizer

public enum EdgeToolsAutoTokenizer {
  #if FoundationEssentials
    public static func from(
      modelDirectory directoryURL: URL
    ) async throws -> sending any EdgeToolsTokenizer {
      let tokenizerJSON = contents(of: directoryURL.appending(path: "tokenizer.json"))
      guard let tokenizerJSON else {
        throw EdgeToolsError.noCompatibleTokenizer(
          in: directoryURL.path(),
          hasHuggingFaceTokenizer: false
        )
      }

      #if HuggingFaceTokenizers && canImport(CTokenizers)
        do {
          return try HuggingFaceTokenizer(
            tokenizerJSON: tokenizerJSON,
            configuration: contents(of: directoryURL.appending(path: "tokenizer_config.json")),
            chatTemplate: try loadChatTemplate(from: directoryURL)
          )
        } catch {
          throw EdgeToolsError.noCompatibleTokenizer(
            in: directoryURL.path(),
            hasHuggingFaceTokenizer: true,
            failures: ["\(error)"]
          )
        }
      #else
        throw EdgeToolsError.noCompatibleTokenizer(
          in: directoryURL.path(),
          hasHuggingFaceTokenizer: true
        )
      #endif
    }
  #endif
}

// MARK: - No Compatible Tokenizer

extension EdgeToolsError {
  static func noCompatibleTokenizer(
    in directory: String,
    hasHuggingFaceTokenizer: Bool,
    failures: [String] = []
  ) -> Self {
    var details = [String]()

    #if HuggingFaceTokenizers && canImport(CTokenizers)
      if hasHuggingFaceTokenizer {
        details.append("tokenizer.json is not a supported Hugging Face tokenizer.")
      } else {
        details.append("tokenizer.json was not found.")
      }
    #else
      if hasHuggingFaceTokenizer {
        details.append(
          "tokenizer.json exists, but the HuggingFaceTokenizers trait is not enabled."
        )
      } else {
        details.append(
          "The HuggingFaceTokenizers trait is not enabled; enable it to search for tokenizer.json."
        )
      }
    #endif

    details.append(contentsOf: failures)
    return Self(
      code: .noCompatibleTokenizer,
      message:
        "No compatible tokenizer was found in \(directory). \(details.joined(separator: " "))"
    )
  }
}

// MARK: - Helpers

#if FoundationEssentials
  private func contents(of fileURL: URL) -> Data? {
    try? Data(contentsOf: fileURL)
  }
#endif

#if HuggingFaceTokenizers && FoundationEssentials && canImport(CTokenizers)
  private func loadChatTemplate(from directoryURL: URL) throws -> String? {
    if let jinja = contents(of: directoryURL.appending(path: "chat_template.jinja")) {
      return String(decoding: jinja, as: UTF8.self)
    }
    guard let json = contents(of: directoryURL.appending(path: "chat_template.json")) else {
      return nil
    }
    return try JSONDecoder().decode(ChatTemplateFile.self, from: json).chatTemplate
  }

  private struct ChatTemplateFile: Decodable {
    let chatTemplate: String

    enum CodingKeys: String, CodingKey {
      case chatTemplate = "chat_template"
    }
  }
#endif
