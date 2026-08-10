import ArgumentParser
import EdgeTools
import Foundation

// MARK: - GrammarOption

public enum GrammarOption: Hashable, Sendable {
  // The model's own tool call grammar.
  case auto
  case unconstrained
  case custom(format: GrammarFormat, source: GrammarSourceLocation)
}

// MARK: - GrammarFormat

public enum GrammarFormat: String, Hashable, Sendable, CaseIterable {
  case ebnf
  case lark
  case regex
  case jsonSchema = "json-schema"
  case structuralTag = "structural-tag"
  case serialized

  fileprivate init?(pathExtension: String) {
    switch pathExtension.lowercased() {
    case "ebnf", "gbnf": self = .ebnf
    case "lark": self = .lark
    case "json": self = .jsonSchema
    default: return nil
    }
  }
}

// MARK: - GrammarSourceLocation

public enum GrammarSourceLocation: Hashable, Sendable {
  case inline(String)
  case file(URL)
}

// MARK: - Parsing

extension GrammarOption: ExpressibleByArgument {
  public init?(argument: String) {
    switch argument {
    case "auto":
      self = .auto
    case "unconstrained":
      self = .unconstrained
    default:
      let separator = argument.firstIndex(of: ":")
      let prefix = separator.map { String(argument[argument.startIndex..<$0]) }
      if let prefix, let format = GrammarFormat(rawValue: prefix), let separator {
        self = .custom(
          format: format,
          source: .inline(String(argument[argument.index(after: separator)...]))
        )
      } else {
        let url = URL(fileURLWithPath: argument)
        guard let format = GrammarFormat(pathExtension: url.pathExtension) else { return nil }
        self = .custom(format: format, source: .file(url))
      }
    }
  }

  public static var allValueStrings: [String] {
    ["auto", "unconstrained"] + GrammarFormat.allCases.map { "\($0.rawValue):<inline>" }
  }
}

// MARK: - Constraints

extension GrammarOption {
  func validate() throws {
    guard case .custom(let format, let source) = self else { return }
    _ = try grammar(format: format, source: source)
  }

  public func constraint(toolCallRange: GrammarToolCallRange) throws -> XGRGenerationConstraint {
    switch self {
    case .auto:
      return .toolsWithGrammar(range: toolCallRange)
    case .unconstrained:
      return .unconstrained
    case .custom(let format, let source):
      return .grammar(try grammar(format: format, source: source))
    }
  }
}

private func grammar(format: GrammarFormat, source: GrammarSourceLocation) throws -> XGRGrammar {
  let text: String
  switch source {
  case .inline(let value):
    text = value
  case .file(let url):
    guard let contents = try? String(contentsOf: url, encoding: .utf8) else {
      throw EdgeCLIError("Could not read grammar file \(url.path()).")
    }
    text = contents
  }
  do {
    switch format {
    case .ebnf: return try .ebnf(text)
    case .lark: return try .lark(text)
    case .regex: return try .regex(text)
    case .jsonSchema: return try .jsonSchema(text)
    case .structuralTag: return try .structuralTagJSON(text)
    case .serialized: return try .serializedJSON(text)
    }
  } catch {
    throw EdgeCLIError("Invalid \(format.rawValue) grammar: \(error)")
  }
}
