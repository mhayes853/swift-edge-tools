import EdgeToolsMacros
import SwiftParser
import SwiftSyntax
import SwiftSyntaxMacroExpansion
import Testing

extension `EdgeToolsMacros tests` {
  @Suite
  struct `EdgeToolsStreamGenerableMacro tests` {
    @Test
    func `Generates Stream Parsing For A Struct Without An Argument`() throws {
      let expansion = try streamExpansion(
        """
        @EdgeToolsGenerable
        struct SearchRequest {
          @EdgeToolsGuide(key: "q")
          var query: String
        }
        """
      )

      #expect(
        expansion.contains("extension SearchRequest: EdgeToolsGenerable, StreamParsingCore.StreamParseable")
      )
      #expect(expansion.contains("struct Partial: StreamParsingCore.StreamParseable"))
      #expect(expansion.contains("key: \"q\""))
      #expect(expansion.contains("query: String.Partial?"))
      #expect(expansion.contains("String.Partial?.edgeToolsGenerationSchema"))
      #expect(expansion.contains("init?(streamPartial partial: Partial)"))
    }

    @Test
    func `Generates Stream Parsing For A Defaulted Enum`() throws {
      let expansion = try streamExpansion(
        """
        @EdgeToolsGenerable
        enum Action {
          @StreamParseableDefault
          case idle(reason: String?)
          case search(query: String)
        }
        """
      )

      #expect(
        expansion.contains("extension Action: EdgeToolsGenerable, StreamParsingCore.StreamParseable")
      )
      #expect(expansion.contains("enum SearchPayload"))
      #expect(expansion.contains("String.Partial?.edgeToolsGenerationSchema"))
      #expect(expansion.contains("init?(streamPartial partial: Partial)"))
    }

    @Test
    func `Preserves A User Declared Partial`() throws {
      let expansion = try streamExpansion(
        """
        @EdgeToolsGenerable
        struct SearchRequest {
          var query: String
          struct Partial {}
        }
        """
      )

      #expect(expansion == "extension SearchRequest: EdgeToolsGenerable {}")
    }
  }
}

private func streamExpansion(_ source: String) throws -> String {
  let declaration = try #require(Parser.parse(source: source).statements.first?.item.as(DeclSyntax.self))
  let context = BasicMacroExpansionContext()
  if let structDeclaration = declaration.as(StructDeclSyntax.self) {
    let attribute = try #require(structDeclaration.attributes.first?.as(AttributeSyntax.self))
    return try EdgeToolsGenerableMacro.expansion(
      of: attribute,
      attachedTo: structDeclaration,
      providingExtensionsOf: IdentifierTypeSyntax(name: structDeclaration.name),
      conformingTo: [],
      in: context
    ).map(\.trimmedDescription).joined(separator: "\n")
  }
  let enumDeclaration = try #require(declaration.as(EnumDeclSyntax.self))
  let attribute = try #require(enumDeclaration.attributes.first?.as(AttributeSyntax.self))
  return try EdgeToolsGenerableMacro.expansion(
    of: attribute,
    attachedTo: enumDeclaration,
    providingExtensionsOf: IdentifierTypeSyntax(name: enumDeclaration.name),
    conformingTo: [],
    in: context
  ).map(\.trimmedDescription).joined(separator: "\n")
}
