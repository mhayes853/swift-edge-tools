import Foundation
import SwiftDiagnostics
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

public enum EdgeToolsGenerableMacro: ExtensionMacro, MemberMacro {
  public static func expansion(
    of node: AttributeSyntax,
    providingMembersOf declaration: some DeclGroupSyntax,
    conformingTo protocols: [TypeSyntax],
    in context: some MacroExpansionContext
  ) throws -> [DeclSyntax] {
    let schemaFragments = Self.schemaFragments(from: node, context: context)
    let accessModifier = Self.accessModifier(for: declaration)
    let modifierPrefix = Self.modifierPrefix(for: accessModifier)
    var members = [DeclSyntax]()

    if let structDecl = declaration.as(StructDeclSyntax.self) {
      let properties = Self.storedProperties(in: structDecl, context: context)
      if !Self.hasExistingEdgeToolsGenerationSchema(in: declaration) {
        members.append(
          Self.generationSchemaProperty(
            from: properties,
            modifierPrefix: modifierPrefix,
            schemaFragments: schemaFragments
          )
        )
      }

      if !Self.hasExistingEdgeToolsValueInitializer(in: declaration) {
        members.append(Self.valueInitializer(from: properties, modifierPrefix: modifierPrefix))
      }

      if !Self.hasExistingEdgeToolsValueProperty(in: declaration) {
        members.append(Self.valueProperty(from: properties, modifierPrefix: modifierPrefix))
      }
    } else if let enumDecl = declaration.as(EnumDeclSyntax.self) {
      let cases = try Self.enumCases(in: enumDecl)
      if !Self.hasExistingEdgeToolsGenerationSchema(in: declaration) {
        members.append(
          Self.enumGenerationSchemaProperty(
            from: cases,
            modifierPrefix: modifierPrefix,
            schemaFragments: schemaFragments
          )
        )
      }

      if !Self.hasExistingEdgeToolsValueInitializer(in: declaration) {
        members.append(
          Self.enumValueInitializer(
            typeName: enumDecl.name.text,
            cases: cases,
            modifierPrefix: modifierPrefix
          )
        )
      }

      if !Self.hasExistingEdgeToolsValueProperty(in: declaration) {
        members.append(Self.enumValueProperty(from: cases, modifierPrefix: modifierPrefix))
      }
    } else {
      throw MacroExpansionErrorMessage(
        "@EdgeToolsGenerable can only be applied to struct or enum declarations."
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
    guard declaration.is(StructDeclSyntax.self) || declaration.is(EnumDeclSyntax.self) else {
      throw MacroExpansionErrorMessage(
        "@EdgeToolsGenerable can only be applied to struct or enum declarations."
      )
    }
    let typeName = type.trimmedDescription
    return [
      try ExtensionDeclSyntax(
        """
        extension \(raw: typeName): EdgeToolsGenerable {}
        """
      )
    ]
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

  private struct AssociatedValue {
    let sourceLabel: String?
    let schemaKey: String
    let typeName: String
    let isOptional: Bool
    let bindingName: String
  }

  private struct EnumCase {
    let name: String
    let sourceName: String
    let associatedValues: [AssociatedValue]
  }

  private static func hasExistingEdgeToolsGenerationSchema(
    in declaration: some DeclGroupSyntax
  ) -> Bool {
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

  private static func hasExistingEdgeToolsValueInitializer(
    in declaration: some DeclGroupSyntax
  ) -> Bool {
    declaration.memberBlock.members.contains { member in
      guard let initializer = member.decl.as(InitializerDeclSyntax.self) else { return false }
      let parameters = initializer.signature.parameterClause.parameters
      guard parameters.count == 1, let parameter = parameters.first else { return false }
      return parameter.firstName.text == "edgeToolsValue"
    }
  }

  private static func hasExistingEdgeToolsValueProperty(
    in declaration: some DeclGroupSyntax
  ) -> Bool {
    declaration.memberBlock.members.contains { member in
      guard let variableDecl = member.decl.as(VariableDeclSyntax.self) else { return false }
      guard !Self.isStatic(variableDecl) else { return false }
      return variableDecl.bindings.contains { binding in
        guard let identifierPattern = binding.pattern.as(IdentifierPatternSyntax.self) else {
          return false
        }
        return identifierPattern.identifier.text == "edgeToolsValue"
      }
    }
  }

  private static func isStatic(_ variableDecl: VariableDeclSyntax) -> Bool {
    variableDecl.modifiers.contains { $0.name.tokenKind == .keyword(.static) }
  }

  private static func accessModifier(for declaration: some DeclGroupSyntax) -> String? {
    declaration.modifiers
      .first { modifier in
        switch modifier.name.tokenKind {
        case .keyword(.public), .keyword(.package), .keyword(.fileprivate), .keyword(.private):
          true
        default:
          false
        }
      }
      .map { modifier in
        switch modifier.name.tokenKind {
        case .keyword(.public):
          "public"
        case .keyword(.package):
          "package"
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
      guard Self.isStoredProperty(binding) else { return nil }
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
      let guideAttributes = Self.guideAttributes(in: variableDecl)
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
      let ignoredAttribute = Self.ignoredAttribute(in: variableDecl)
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

  private static func enumCases(
    in declaration: EnumDeclSyntax
  ) throws -> [EnumCase] {
    let elements = declaration.memberBlock.members.flatMap { member -> [EnumCaseElementSyntax] in
      member.decl.as(EnumCaseDeclSyntax.self).map { Array($0.elements) } ?? []
    }
    guard !elements.isEmpty else {
      throw MacroExpansionErrorMessage(
        "@EdgeToolsGenerable enums must declare at least one case."
      )
    }

    var caseNames = Set<String>()
    return try elements.map { element in
      let name = element.name.text
      guard caseNames.insert(name).inserted else {
        throw MacroExpansionErrorMessage(
          "@EdgeToolsGenerable does not support overloaded enum case names ('\(name)')."
        )
      }
      guard let parameters = element.parameterClause?.parameters, !parameters.isEmpty else {
        throw MacroExpansionErrorMessage(
          "@EdgeToolsGenerable enum case '\(name)' must have at least one associated value."
        )
      }

      var schemaKeys = Set<String>()
      let associatedValues = try parameters.enumerated().map { index, parameter in
        let rawLabel = parameter.firstName?.text
        let label = rawLabel == "_" ? nil : rawLabel
        let schemaKey = label ?? "_\(index)"
        guard schemaKeys.insert(schemaKey).inserted else {
          throw MacroExpansionErrorMessage(
            "Enum case '\(name)' has multiple associated values represented by the key '\(schemaKey)'."
          )
        }
        return AssociatedValue(
          sourceLabel: label == nil ? nil : parameter.firstName?.trimmedDescription,
          schemaKey: schemaKey,
          typeName: parameter.type.trimmedDescription,
          isOptional: Self.isOptionalTypeName(parameter.type.trimmedDescription),
          bindingName: "value\(index)"
        )
      }

      return EnumCase(
        name: name,
        sourceName: element.name.trimmedDescription,
        associatedValues: associatedValues
      )
    }
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

  private static func generationSchemaProperty(
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

  private static func enumGenerationSchemaProperty(
    from cases: [EnumCase],
    modifierPrefix: String,
    schemaFragments: [String]
  ) -> DeclSyntax {
    let choices = cases.map { enumCase in
      let propertyPairs = enumCase.associatedValues.map { value in
        "\(Self.quotedStringLiteral(value.schemaKey)): \(value.typeName).edgeToolsGenerationSchema"
      }
      .joined(separator: ",\n                      ")
      let required = enumCase.associatedValues.filter { !$0.isOptional }
        .map { Self.quotedStringLiteral($0.schemaKey) }
        .joined(separator: ", ")
      if required.isEmpty {
        return """
          EdgeToolsGenerationSchema(
            .type(.object),
            .properties([
              \(Self.quotedStringLiteral(enumCase.name)): EdgeToolsGenerationSchema(
                .type(.object),
                .properties([
                  \(propertyPairs)
                ]),
                .additionalProperties(false)
              )
            ]),
            .required([\(Self.quotedStringLiteral(enumCase.name))]),
            .additionalProperties(false)
          )
          """
      }
      return """
        EdgeToolsGenerationSchema(
          .type(.object),
          .properties([
            \(Self.quotedStringLiteral(enumCase.name)): EdgeToolsGenerationSchema(
              .type(.object),
              .properties([
                \(propertyPairs)
              ]),
              .required([\(required)]),
              .additionalProperties(false)
            )
          ]),
          .required([\(Self.quotedStringLiteral(enumCase.name))]),
          .additionalProperties(false)
        )
        """
    }
    var fragments = [".anyOf([\n            \(choices.joined(separator: ",\n            "))\n          ])"]
    fragments.append(contentsOf: schemaFragments)
    return """
      \(raw: modifierPrefix)static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
        EdgeToolsGenerationSchema(
          \(raw: fragments.joined(separator: ",\n          "))
        )
      }
      """
  }

  private static func valueInitializer(
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

  private static func enumValueInitializer(
    typeName: String,
    cases: [EnumCase],
    modifierPrefix: String
  ) -> DeclSyntax {
    let caseInitializers = cases.map { enumCase in
      let keys = enumCase.associatedValues.filter { !$0.isOptional }
        .map { Self.quotedStringLiteral($0.schemaKey) }
        .joined(separator: ", ")
      let arguments = enumCase.associatedValues.map { value in
        let expression =
          "try \(Self.initializerTypeName(for: value.typeName))(edgeToolsValue: _edgeToolsValue(payload, forKey: \(Self.quotedStringLiteral(value.schemaKey))))"
        return value.sourceLabel.map { "\($0): \(expression)" } ?? expression
      }
      .joined(separator: ",\n          ")
      return """
        if let value = object[\(Self.quotedStringLiteral(enumCase.name))] {
          let payload = try _edgeToolsRequireObjectValue(value, keys: [\(keys)])
          self = .\(enumCase.sourceName)(
            \(arguments)
          )
          return
        }
        """
    }
    .joined(separator: "\n")
    return """
      \(raw: modifierPrefix)init(edgeToolsValue: EdgeToolsValue) throws {
        let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
        \(raw: caseInitializers)
        throw EdgeToolsUnknownEnumCaseError(
          typeName: \(raw: Self.quotedStringLiteral(typeName)),
          caseName: object.keys.first ?? ""
        )
      }
      """
  }

  private static func valueProperty(
    from properties: [StoredProperty],
    modifierPrefix: String
  ) -> DeclSyntax {
    let entries =
      properties.compactMap { property -> String? in
        guard !property.isIgnored else { return nil }
        let valueExpression =
          property.isOptional
          ? "self.\(property.name)?.edgeToolsValue"
          : "self.\(property.name).edgeToolsValue"
        return
          "(key: \(Self.quotedStringLiteral(property.schemaKey)), value: \(valueExpression))"
      }

    if entries.isEmpty {
      return """
        \(raw: modifierPrefix)var edgeToolsValue: EdgeToolsValue {
          _edgeToolsBuildObjectValue()
        }
        """
    }

    return """
      \(raw: modifierPrefix)var edgeToolsValue: EdgeToolsValue {
        _edgeToolsBuildObjectValue(
          \(raw: entries.joined(separator: ",\n          "))
        )
      }
      """
  }

  private static func enumValueProperty(
    from cases: [EnumCase],
    modifierPrefix: String
  ) -> DeclSyntax {
    let switchCases = cases.map { enumCase in
      let bindings = enumCase.associatedValues.map { "let \($0.bindingName)" }
        .joined(separator: ", ")
      let entries = enumCase.associatedValues.map { value in
        "(key: \(Self.quotedStringLiteral(value.schemaKey)), value: \(value.bindingName).edgeToolsValue)"
      }
      .joined(separator: ",\n            ")
      return """
        case .\(enumCase.sourceName)(\(bindings)):
          _edgeToolsBuildObjectValue(
            (key: \(Self.quotedStringLiteral(enumCase.name)), value: _edgeToolsBuildObjectValue(
              \(entries)
            )))
        """
    }
    .joined(separator: "\n")
    return """
      \(raw: modifierPrefix)var edgeToolsValue: EdgeToolsValue {
        switch self {
        \(raw: switchCases)
        }
      }
      """
  }

  private static func guideAttributes(in variableDecl: VariableDeclSyntax) -> [AttributeSyntax] {
    Self.attributes(named: ["EdgeToolsGuide", "EdgeTools.EdgeToolsGuide"], in: variableDecl)
  }

  private static func ignoredAttribute(in variableDecl: VariableDeclSyntax) -> AttributeSyntax? {
    Self.attributes(named: ["EdgeToolsIgnored", "EdgeTools.EdgeToolsIgnored"], in: variableDecl)
      .first
  }

  private static func attributes(
    named names: Set<String>,
    in variableDecl: VariableDeclSyntax
  ) -> [AttributeSyntax] {
    variableDecl.attributes.compactMap { element in
      guard let attribute = element.as(AttributeSyntax.self) else { return nil }
      return names.contains(attribute.attributeName.trimmedDescription) ? attribute : nil
    }
  }

  private static func isStoredProperty(_ binding: PatternBindingSyntax) -> Bool {
    guard let accessorBlock = binding.accessorBlock else { return true }
    switch accessorBlock.accessors {
    case .accessors(let accessors):
      return accessors.allSatisfy { accessor in
        switch accessor.accessorSpecifier.tokenKind {
        case .keyword(.willSet), .keyword(.didSet): true
        default: false
        }
      }
    case .getter:
      return false
    }
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
    expression.as(StringLiteralExprSyntax.self)?.representedLiteralValue
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
