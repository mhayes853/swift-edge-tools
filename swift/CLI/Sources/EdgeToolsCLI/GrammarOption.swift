import ArgumentParser
import EdgeTools
import Foundation

// MARK: - GrammarOption

/// The generation constraint applied while decoding.
public enum GrammarOption: Hashable, Sendable {
  /// The model's own tool call grammar.
  case auto
  case unconstrained
  case builtinJSON
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
    case "json":
      self = .builtinJSON
    default:
      let separator = argument.firstIndex(of: ":")
      let prefix = separator.map { String(argument[argument.startIndex..<$0]) }
      if let prefix, let format = GrammarFormat(rawValue: prefix), let separator {
        let value = String(argument[argument.index(after: separator)...])
        self = .custom(format: format, source: .inline(value))
      } else {
        let url = URL(fileURLWithPath: argument)
        guard let format = GrammarFormat(pathExtension: url.pathExtension) else { return nil }
        self = .custom(format: format, source: .file(url))
      }
    }
  }

  public static var allValueStrings: [String] {
    ["auto", "unconstrained", "json"] + GrammarFormat.allCases.map { "\($0.rawValue):<inline>" }
  }
}

extension GrammarFormat {
  init?(pathExtension: String) {
    switch pathExtension.lowercased() {
    case "ebnf", "gbnf": self = .ebnf
    case "lark": self = .lark
    case "json": self = .jsonSchema
    default: return nil
    }
  }
}

// MARK: - Constraint

extension GrammarOption {
  public func constraint(
    toolCallRange: GrammarToolCallRange
  ) throws -> EdgeToolsXGRGenerationConstraint {
    switch self {
    case .auto:
      return .toolsWithGrammar(range: toolCallRange)
    case .unconstrained:
      return .unconstrained
    case .builtinJSON:
      return .grammar(.builtinJSONGrammar())
    case .custom(let format, let source):
      return .grammar(try makeGrammar(format: format, source: source))
    }
  }
}

private func makeGrammar(
  format: GrammarFormat,
  source: GrammarSourceLocation
) throws -> XGRGrammar {
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
