import SwiftDiagnostics
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

public enum EdgeToolsGenerableMacro: ExtensionMacro, MemberMacro, MemberAttributeMacro {
  public static func expansion(
    of node: AttributeSyntax,
    providingMembersOf declaration: some DeclGroupSyntax,
    conformingTo protocols: [TypeSyntax],
    in context: some MacroExpansionContext
  ) throws -> [DeclSyntax] {
    let structDecl = try Self.requireStructDecl(declaration: declaration)
    let properties = Self.storedProperties(in: structDecl, context: context)
    let schemaFragments = Self.schemaFragments(from: node, context: context)
    let accessModifier = Self.accessModifier(for: structDecl)
    let modifierPrefix = Self.modifierPrefix(for: accessModifier)
    var members = [DeclSyntax]()

    if !Self.hasExistingEdgeToolsGenerationSchema(in: structDecl) {
      members.append(
        Self.edgeToolsGenerationSchemaProperty(
          from: properties,
          modifierPrefix: modifierPrefix,
          schemaFragments: schemaFragments
        )
      )
    }

    if !Self.hasExistingEdgeToolsValueInitializer(in: structDecl) {
      members.append(
        Self.edgeToolsValueInitializer(
          from: properties,
          modifierPrefix: modifierPrefix
        )
      )
    }

    return members
  }

  public static func expansion(
    of node: AttributeSyntax,
    attachedTo declaration: some DeclGroupSyntax,
    providingExtensionsOf type: some TypeSyntaxProtocol,
    conformingTo protocols: [TypeSyntax],
    in context: some MacroExpansionContext
  ) throws -> [ExtensionDeclSyntax] {
    _ = try Self.requireStructDecl(declaration: declaration)
    let typeName = type.trimmedDescription
    return [
      try ExtensionDeclSyntax(
        """
        extension \(raw: typeName): EdgeToolsGenerable {}
        """
      )
    ]
  }

  public static func expansion(
    of node: AttributeSyntax,
    attachedTo declaration: some DeclGroupSyntax,
    providingAttributesFor member: some DeclSyntaxProtocol,
    in context: some MacroExpansionContext
  ) throws -> [AttributeSyntax] {
    []
  }
}

extension EdgeToolsGenerableMacro {
  private struct StoredProperty {
    let name: String
    let schemaKey: String
    let typeName: String
    let initializerTypeName: String
    let isIgnored: Bool
    let isOptional: Bool
    let hasDefaultValue: Bool
    let schemaExpression: String
  }

  private struct EdgeToolsGuideSelection {
    let key: String?
    let schemaFragments: [String]
  }

  private static func requireStructDecl(
    declaration: some DeclGroupSyntax
  ) throws -> StructDeclSyntax {
    guard let structDecl = declaration.as(StructDeclSyntax.self) else {
      throw MacroExpansionErrorMessage(
        "@EdgeToolsGenerable can only be applied to struct declarations."
      )
    }
    return structDecl
  }

  private static func hasExistingEdgeToolsGenerationSchema(in declaration: StructDeclSyntax) -> Bool
  {
    declaration.memberBlock.members.contains { member in
      guard let variableDecl = member.decl.as(VariableDeclSyntax.self) else { return false }
      guard Self.isStatic(variableDecl) else { return false }
      return variableDecl.bindings.contains { binding in
        guard let identifierPattern = binding.pattern.as(IdentifierPatternSyntax.self) else {
          return false
        }
        return identifierPattern.identifier.text == "edgeToolsGenerationSchema"
      }
    }
  }

  private static func hasExistingEdgeToolsValueInitializer(in declaration: StructDeclSyntax) -> Bool
  {
    declaration.memberBlock.members.contains { member in
      guard let initializer = member.decl.as(InitializerDeclSyntax.self) else { return false }
      let parameters = initializer.signature.parameterClause.parameters
      guard parameters.count == 1, let parameter = parameters.first else { return false }
      return parameter.firstName.text == "edgeToolsValue"
    }
  }

  private static func isStatic(_ variableDecl: VariableDeclSyntax) -> Bool {
    variableDecl.modifiers.contains { $0.name.tokenKind == .keyword(.static) }
  }

  private static func accessModifier(for declaration: StructDeclSyntax) -> String? {
    declaration.modifiers
      .first { modifier in
        switch modifier.name.tokenKind {
        case .keyword(.public), .keyword(.fileprivate), .keyword(.private):
          true
        default:
          false
        }
      }
      .map { modifier in
        switch modifier.name.tokenKind {
        case .keyword(.public):
          "public"
        case .keyword(.fileprivate):
          "fileprivate"
        case .keyword(.private):
          ""
        default:
          ""
        }
      }
      .flatMap { $0.isEmpty ? nil : $0 }
  }

  private static func modifierPrefix(for accessModifier: String?) -> String {
    accessModifier.map { "\($0) " } ?? ""
  }

  private static func storedProperties(
    in declaration: StructDeclSyntax,
    context: some MacroExpansionContext
  ) -> [StoredProperty] {
    declaration.memberBlock.members.reduce(into: [StoredProperty]()) { properties, member in
      guard let variableDecl = member.decl.as(VariableDeclSyntax.self) else { return }
      guard !Self.isStatic(variableDecl) else { return }
      properties.append(contentsOf: Self.storedProperties(from: variableDecl, context: context))
    }
  }

  private static func storedProperties(
    from variableDecl: VariableDeclSyntax,
    context: some MacroExpansionContext
  ) -> [StoredProperty] {
    variableDecl.bindings.compactMap { binding in
      guard binding.accessorBlock == nil else { return nil }
      guard let identifierPattern = binding.pattern.as(IdentifierPatternSyntax.self) else {
        return nil
      }
      guard let type = binding.typeAnnotation?.type else {
        context.diagnose(
          Diagnostic(
            node: Syntax(binding),
            message: SimpleDiagnostic("Stored properties must declare an explicit type.")
          )
        )
        return nil
      }
      let propertyName = identifierPattern.identifier.text
      let typeName = type.trimmedDescription
      let guideAttributes = Self.edgeToolsGuideAttributes(in: variableDecl)
      if guideAttributes.count > 1 {
        context.diagnose(
          Diagnostic(
            node: Syntax(variableDecl),
            message: SimpleDiagnostic(
              "Only one @EdgeToolsGuide attribute can be applied to a stored property."
            )
          )
        )
      }
      let guideSelection = guideAttributes.first.flatMap { attribute in
        Self.parseEdgeToolsGuide(in: attribute, context: context)
      }
      let ignoredAttribute = Self.edgeToolsIgnoredAttribute(in: variableDecl)
      var isIgnored = ignoredAttribute != nil
      let hasDefaultValue = binding.initializer != nil
      let isOptional = Self.isOptionalTypeName(typeName)

      if isIgnored && guideSelection != nil {
        context.diagnose(
          Diagnostic(
            node: Syntax(variableDecl),
            message: SimpleDiagnostic(
              "@EdgeToolsIgnored cannot be combined with @EdgeToolsGuide on the same property."
            )
          )
        )
        isIgnored = false
      }

      if isIgnored && !isOptional && !hasDefaultValue {
        context.diagnose(
          Diagnostic(
            node: Syntax(variableDecl),
            message: SimpleDiagnostic(
              "@EdgeToolsIgnored requires an optional property or a default value."
            )
          )
        )
        isIgnored = false
      }

      let schemaKey = guideSelection?.key ?? propertyName
      let schemaExpression = Self.schemaExpression(
        typeName: typeName,
        guideSelection: guideSelection
      )

      return StoredProperty(
        name: propertyName,
        schemaKey: schemaKey,
        typeName: typeName,
        initializerTypeName: Self.initializerTypeName(for: typeName),
        isIgnored: isIgnored,
        isOptional: isOptional,
        hasDefaultValue: hasDefaultValue,
        schemaExpression: schemaExpression
      )
    }
  }

  private static func schemaExpression(
    typeName: String,
    guideSelection: EdgeToolsGuideSelection?
  ) -> String {
    let baseExpression = "\(typeName).edgeToolsGenerationSchema"
    guard let guideSelection else { return baseExpression }
    guard !guideSelection.schemaFragments.isEmpty else { return baseExpression }
    let fragments = ([baseExpression] + guideSelection.schemaFragments).joined(separator: ", ")
    return "EdgeToolsGenerationSchema(\(fragments))"
  }

  private static func schemaFragments(
    from attribute: AttributeSyntax,
    context: some MacroExpansionContext
  ) -> [String] {
    guard case .argumentList(let arguments) = attribute.arguments else { return [] }
    return arguments.compactMap { argument in
      guard argument.label == nil else { return nil }
      return argument.expression.trimmedDescription
    }
  }

  private static func parseEdgeToolsGuide(
    in attribute: AttributeSyntax,
    context: some MacroExpansionContext
  ) -> EdgeToolsGuideSelection? {
    guard case .argumentList(let arguments) = attribute.arguments else {
      return EdgeToolsGuideSelection(key: nil, schemaFragments: [])
    }

    var key: String?
    var fragments = [String]()

    for argument in arguments {
      if let label = argument.label?.text {
        switch label {
        case "key":
          guard let value = Self.stringLiteralValue(from: argument.expression) else {
            context.diagnose(
              Diagnostic(
                node: Syntax(argument),
                message: SimpleDiagnostic("key must be a string literal.")
              )
            )
            continue
          }
          key = value
        default:
          continue
        }
      } else {
        fragments.append(argument.expression.trimmedDescription)
      }
    }

    return EdgeToolsGuideSelection(key: key, schemaFragments: fragments)
  }

  private static func edgeToolsGenerationSchemaProperty(
    from properties: [StoredProperty],
    modifierPrefix: String,
    schemaFragments: [String]
  ) -> DeclSyntax {
    let activeProperties = properties.filter { !$0.isIgnored }
    let propertyPairs =
      activeProperties.map { property in
        "\(Self.quotedStringLiteral(property.schemaKey)): \(property.schemaExpression)"
      }
      .joined(separator: ",\n          ")
    let requiredProperties = activeProperties.filter { !$0.isOptional }
      .map { property in
        Self.quotedStringLiteral(property.schemaKey)
      }
      .joined(separator: ", ")

    var fragments = [".type(.object)"]
    fragments.append(contentsOf: schemaFragments)
    if activeProperties.isEmpty {
      return """
        \(raw: modifierPrefix)static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
          EdgeToolsGenerationSchema(
            \(raw: fragments.joined(separator: ",\n            "))
          )
        }
        """
    }

    fragments.append(
      ".properties([\n                \(propertyPairs)\n              ])"
    )
    if !requiredProperties.isEmpty {
      fragments.append(".required([\(requiredProperties)])")
    }

    return """
      \(raw: modifierPrefix)static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
        EdgeToolsGenerationSchema(
          \(raw: fragments.joined(separator: ",\n          "))
        )
      }
      """
  }

  private static func edgeToolsValueInitializer(
    from properties: [StoredProperty],
    modifierPrefix: String
  ) -> DeclSyntax {
    let assignments =
      properties.compactMap { property -> String? in
        if property.isIgnored {
          if property.hasDefaultValue {
            return nil
          }
          return "self.\(property.name) = nil"
        }
        return
          "self.\(property.name) = try \(property.initializerTypeName)(edgeToolsValue: _edgeToolsValue(object, forKey: \(Self.quotedStringLiteral(property.schemaKey))))"
      }
      .joined(separator: "\n")

    if assignments.isEmpty {
      return """
        \(raw: modifierPrefix)init(edgeToolsValue: EdgeToolsValue) throws {
          _ = try _edgeToolsRequireObjectValue(edgeToolsValue)
        }
        """
    }

    return """
      \(raw: modifierPrefix)init(edgeToolsValue: EdgeToolsValue) throws {
        let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
        \(raw: assignments)
      }
      """
  }

  private static func edgeToolsGuideAttributes(in variableDecl: VariableDeclSyntax)
    -> [AttributeSyntax]
  {
    variableDecl.attributes.compactMap { element in
      guard let attribute = element.as(AttributeSyntax.self) else { return nil }
      guard let identifierType = attribute.attributeName.as(IdentifierTypeSyntax.self) else {
        return nil
      }
      let name = identifierType.name.text
      return name == "EdgeToolsGuide" || name == "EdgeTools.EdgeToolsGuide" ? attribute : nil
    }
  }

  private static func edgeToolsIgnoredAttribute(in variableDecl: VariableDeclSyntax)
    -> AttributeSyntax?
  {
    variableDecl.attributes
      .compactMap { element in
        guard let attribute = element.as(AttributeSyntax.self) else { return nil }
        guard let identifierType = attribute.attributeName.as(IdentifierTypeSyntax.self) else {
          return nil
        }
        let name = identifierType.name.text
        return name == "EdgeToolsIgnored" || name == "EdgeTools.EdgeToolsIgnored" ? attribute : nil
      }
      .first
  }

  private static func isOptionalTypeName(_ typeName: String) -> Bool {
    typeName.hasSuffix("?") || typeName.hasPrefix("Optional<")
      || typeName.hasPrefix("Swift.Optional<")
  }

  private static func initializerTypeName(for typeName: String) -> String {
    let trimmed = typeName.replacingOccurrences(of: " ", with: "")
    if trimmed.hasSuffix("?") {
      return "Optional<\(String(trimmed.dropLast()))>"
    }
    return trimmed
  }

  private static func quotedStringLiteral(_ value: String) -> String {
    let escaped =
      value
      .replacingOccurrences(of: "\\", with: "\\\\")
      .replacingOccurrences(of: "\"", with: "\\\"")
      .replacingOccurrences(of: "\n", with: "\\n")
    return "\"\(escaped)\""
  }

  private static func stringLiteralValue(from expression: ExprSyntax) -> String? {
    let text = expression.trimmedDescription
    guard text.count >= 2, text.first == "\"", text.last == "\"" else { return nil }
    return String(text.dropFirst().dropLast())
  }
}

private struct SimpleDiagnostic: DiagnosticMessage {
  let message: String
  let diagnosticID = MessageID(domain: "EdgeToolsMacros", id: "SimpleDiagnostic")
  let severity = DiagnosticSeverity.error

  init(_ message: String) {
    self.message = message
  }
}
