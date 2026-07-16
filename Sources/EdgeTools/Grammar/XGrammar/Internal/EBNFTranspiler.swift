#if XGrammar
  import Foundation

  struct XGrammarEBNFDocument: Hashable, Sendable {
    struct Rule: Hashable, Sendable {
      let name: String
      var body: String
    }

    var rules: [Rule]

    init(_ source: String) throws {
      var rules = [Rule]()
      for line in source.split(separator: "\n", omittingEmptySubsequences: false) {
        if let separator = line.range(of: "::=") {
          let name = line[..<separator.lowerBound].trimmingCharacters(in: .whitespaces)
          let body = line[separator.upperBound...].trimmingCharacters(in: .whitespaces)
          guard !name.isEmpty else { throw ToolCallXGrammarError.unsupportedSchema }
          rules.append(Rule(name: name, body: body))
        } else if !line.trimmingCharacters(in: .whitespaces).isEmpty {
          guard !rules.isEmpty else { throw ToolCallXGrammarError.unsupportedSchema }
          rules[rules.count - 1].body += "\n" + line
        }
      }
      guard rules.contains(where: { $0.name == "root" }) else {
        throw ToolCallXGrammarError.unsupportedSchema
      }
      self.rules = rules
    }

    var source: String {
      self.rules.map { "\($0.name) ::= \($0.body)" }.joined(separator: "\n")
    }

    mutating func removeDuplicateRules() {
      while true {
        let orderedRules = self.rules.sorted { lhs, _ in lhs.name == "root" }
        var canonicalNames = [String: String]()
        var aliases = [String: String]()
        var uniqueRules = [Rule]()

        for rule in orderedRules {
          if let canonicalName = canonicalNames[rule.body] {
            aliases[rule.name] = canonicalName
          } else {
            canonicalNames[rule.body] = rule.name
            uniqueRules.append(rule)
          }
        }
        guard !aliases.isEmpty else { return }
        for index in uniqueRules.indices {
          uniqueRules[index].body = Self.replacingRuleReferences(
            in: uniqueRules[index].body,
            aliases: aliases
          )
        }
        self.rules = uniqueRules
      }
    }

    mutating func mapLiterals(
      _ transform: (_ ruleName: String, _ value: String, _ suffix: Substring) -> String
    ) throws {
      for index in self.rules.indices {
        let rule = self.rules[index]
        self.rules[index].body = try Self.mapLiterals(in: rule.body) { value, suffix in
          transform(rule.name, value, suffix)
        }
      }
    }

    private static func mapLiterals(
      in source: String,
      _ transform: (_ value: String, _ suffix: Substring) -> String
    ) throws -> String {
      var output = ""
      var outputStart = source.startIndex

      for match in source.matches(of: literalRegex) {
        let literalRange = match.output.1.startIndex..<match.output.1.endIndex
        output.append(contentsOf: source[outputStart..<literalRange.lowerBound])
        let encodedValue = source[literalRange].dropFirst().dropLast()
        let value = Self.decodeLiteral(encodedValue)
        let transformed = transform(value, source[literalRange.upperBound...])
        output.append("\"")
        output.append(contentsOf: Self.escapeLiteral(transformed))
        output.append("\"")
        outputStart = literalRange.upperBound
      }
      output.append(contentsOf: source[outputStart...])
      return output
    }

    private static func replacingRuleReferences(
      in source: String,
      aliases: [String: String]
    ) -> String {
      let literalRanges = source.matches(of: literalRegex).map {
        $0.output.1.startIndex..<$0.output.1.endIndex
      }
      var output = ""
      var outputStart = source.startIndex
      for match in source.matches(of: ruleReferenceRegex) {
        let range = match.range
        let isInsideLiteral = literalRanges.contains {
          $0.lowerBound <= range.lowerBound && range.upperBound <= $0.upperBound
        }
        guard !isInsideLiteral, let replacement = aliases[String(source[range])] else { continue }
        output.append(contentsOf: source[outputStart..<range.lowerBound])
        output.append(replacement)
        outputStart = range.upperBound
      }
      output.append(contentsOf: source[outputStart...])
      return output
    }

    private static func decodeLiteral(_ literal: Substring) -> String {
      var result = ""
      var isEscaping = false
      for character in literal {
        if isEscaping {
          switch character {
          case "n": result.append("\n")
          case "r": result.append("\r")
          case "t": result.append("\t")
          default: result.append(character)
          }
          isEscaping = false
        } else if character == "\\" {
          isEscaping = true
        } else {
          result.append(character)
        }
      }
      return result
    }

    private static func escapeLiteral(_ literal: String) -> String {
      literal.reduce(into: "") { result, character in
        switch character {
        case "\\", "\"":
          result.append("\\")
          result.append(character)
        case "\n": result.append(contentsOf: "\\n")
        case "\r": result.append(contentsOf: "\\r")
        case "\t": result.append(contentsOf: "\\t")
        default: result.append(character)
        }
      }
    }
  }

  extension Substring {
    var hasToolCallContinuationReference: Bool {
      self.contains(toolCallContinuationReferenceRegex)
    }
  }

  nonisolated(unsafe) private let literalRegex = /(?:^|[^\\])(\"(?:\\.|[^\"\\])*\")/

  nonisolated(unsafe) private let toolCallContinuationReferenceRegex =
    /\b(?:root(?:_[A-Za-z0-9]+)*|xml_object(?:_[A-Za-z0-9]+)*)\b/

  nonisolated(unsafe) private let ruleReferenceRegex = /\b[A-Za-z_][A-Za-z0-9_]*\b/
#endif
