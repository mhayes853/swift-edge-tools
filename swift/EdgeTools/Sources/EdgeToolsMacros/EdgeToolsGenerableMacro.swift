import Foundation
import StreamParsingMacroSupport
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
    let schemaFragments = Self.schemaFragments(from: node)
    let accessModifier = Self.accessModifier(for: declaration)
    let modifierPrefix = Self.modifierPrefix(for: accessModifier)
    let partialStrings =
      try Self.argument(named: "partialStrings", in: node)
      .map { try Self.partialStrings(from: $0) } ?? .streamString
    let configuration = Self.streamGenerationConfiguration(
      accessModifier: accessModifier,
      from: node
    )
    var members = [DeclSyntax]()

    if let structDecl = declaration.as(StructDeclSyntax.self) {
      let properties = Self.storedProperties(
        in: structDecl,
        configuration: configuration,
        partialStrings: partialStrings,
        context: context
      )
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
      let cases = try Self.enumCases(
        in: enumDecl,
        configuration: configuration,
        partialStrings: partialStrings
      )
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
    let partialStrings =
      try Self.argument(named: "partialStrings", in: node)
      .map { try Self.partialStrings(from: $0) } ?? .streamString
    let typeName = type.trimmedDescription
    let accessModifier = Self.streamAccessModifier(for: declaration, in: context)
    let configuration = Self.streamGenerationConfiguration(
      accessModifier: accessModifier,
      genericParameters: Self.genericParameters(in: declaration, context: context),
      from: node
    )
    let hasCustomPartial = declaration.memberBlock.members.contains { member in
      member.decl.as(StructDeclSyntax.self)?.name.text == "Partial"
        || member.decl.as(TypeAliasDeclSyntax.self)?.name.text == "Partial"
        || member.decl.as(EnumDeclSyntax.self)?.name.text == "Partial"
    }
    let isGeneric = !configuration.genericParameters.isEmpty
    let basicExtension = try ExtensionDeclSyntax(
      "extension \(raw: typeName): EdgeToolsGenerable {}"
    )
    guard !hasCustomPartial, !declaration.is(EnumDeclSyntax.self) || !isGeneric else {
      return [basicExtension]
    }

    if let structDecl = declaration.as(StructDeclSyntax.self) {
      let properties = Self.storedProperties(
        in: structDecl,
        configuration: configuration,
        partialStrings: partialStrings,
        context: context
      )
      let parsedProperties = properties.filter { !$0.isIgnored }
      let generation = try StreamObjectGeneration(
        fields: parsedProperties.map(\.streamField),
        configuration: configuration
      )
      let partialCustomization = Self.generablePartialCustomization(
        fields: generation.partialFields,
        keyExpressions: parsedProperties.map(\.keyExpression),
        schemaFragments: parsedProperties.map(\.schemaFragments),
        preservesStreamNull: isGeneric,
        accessModifier: accessModifier
      )
      let partial = try generation.structDeclarationSyntax(
        in: context,
        partialCustomization: partialCustomization
      )
      let unparsedMembers = properties.filter { $0.isIgnored && !$0.hasDefaultValue }
        .map { StreamUnparsedMember(name: .identifier($0.name)) }
      let conversions = try generation.conversionsSyntax(
        unparsedMembers: unparsedMembers,
        partialValueInlining: .never
      )
      return [
        try ExtensionDeclSyntax(
          "extension \(raw: typeName): EdgeToolsGenerable, StreamParsingCore.StreamParseable"
        ) {
          partial
          conversions
        }
      ]
    }

    let enumDecl = declaration.as(EnumDeclSyntax.self)!
    guard let defaultCase = try Self.streamDefaultCase(in: enumDecl) else {
      return [basicExtension]
    }
    let cases = try Self.enumCases(
      in: enumDecl,
      configuration: configuration,
      partialStrings: partialStrings
    )
    let generation = try StreamEnumGeneration(
      cases: cases.map { enumCase in
        StreamParseableEnumCase(
          name: enumCase.sourceToken,
          associatedValues: enumCase.associatedValues.map(\.streamField)
        )
      },
      representation: .caseKeyedObject,
      defaultCase: defaultCase,
      configuration: configuration
    )
    guard let partialFields = generation.partialFields else {
      throw MacroExpansionErrorMessage("Stream parsing did not plan an enum Partial.")
    }
    let generablePartials = try generation.partialSyntax(
      in: context,
      partialCustomization: Self.generablePartialCustomization(
        fields: partialFields,
        keyExpressions: cases.map(\.keyExpression),
        accessModifier: accessModifier
      ),
      payloadCustomization: { payload in
        .generated(
          partial: Self.generablePartialCustomization(
            fields: payload.partialFields,
            keyExpressions: payload.fields.map {
              Self.schemaKeyExpression(for: $0, configuration: configuration)
            },
            accessModifier: accessModifier
          )
        )
      }
    )
    let conversions = generation.conversionsSyntax(partialValueInlining: .never)
    return [
      try ExtensionDeclSyntax(
        "extension \(raw: typeName): EdgeToolsGenerable, StreamParsingCore.StreamParseable"
      ) {
        generablePartials
        conversions
      }
    ]
  }
}

extension EdgeToolsGenerableMacro {
  private struct StoredProperty {
    let streamField: StreamParseableField
    let keyExpression: String
    var isIgnored = false
    var schemaFragments = [String]()

    var name: String { self.streamField.name.text }
    var typeName: String { self.streamField.type.trimmedDescription }
    var initializerTypeName: String {
      EdgeToolsGenerableMacro.initializerTypeName(for: self.typeName)
    }
    var isOptional: Bool { self.streamField.type.streamIsOptional }
    var hasDefaultValue: Bool { self.streamField.defaultValue != nil }
    var schemaExpression: String {
      let base: String
      if let conversion = self.streamField.completedConversion {
        base = "\(conversion).Source.edgeToolsGenerationSchema" + (self.isOptional ? ".nullable()" : "")
      } else {
        base = "\(self.typeName).edgeToolsGenerationSchema"
      }
      return self.schemaFragments.isEmpty
        ? base
        : "EdgeToolsGenerationSchema(\(([base] + self.schemaFragments).joined(separator: ", ")))"
    }
  }

  private struct EdgeToolsGuideSelection {
    var key: String?
    var schemaFragments = [String]()
    var initialCapacity: ExprSyntax?
    var partialStrings: StreamPartialStrings?
    var completedConversion: TypeSyntax?
  }

  private struct AssociatedValue {
    let streamField: StreamParseableField
    let keyExpression: String
    let bindingName: String

    var sourceLabel: String? {
      self.streamField.name.tokenKind == .wildcard ? nil : self.streamField.name.trimmedDescription
    }
    var typeName: String { self.streamField.type.trimmedDescription }
    var isOptional: Bool { self.streamField.type.streamIsOptional }
  }

  private struct EnumCase {
    let sourceToken: TokenSyntax
    let keyExpression: String
    let associatedValues: [AssociatedValue]

    var name: String { self.sourceToken.text }
    var sourceName: String { self.sourceToken.trimmedDescription }
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
    configuration: StreamGenerationConfiguration,
    partialStrings: StreamPartialStrings,
    context: some MacroExpansionContext
  ) -> [StoredProperty] {
    declaration.memberBlock.members.flatMap { member -> [StoredProperty] in
      guard let variableDecl = member.decl.as(VariableDeclSyntax.self),
        !Self.isStatic(variableDecl)
      else {
        return []
      }
      return Self.storedProperties(
        from: variableDecl,
        configuration: configuration,
        partialStrings: partialStrings,
        context: context
      )
    }
  }

  private static func storedProperties(
    from variableDecl: VariableDeclSyntax,
    configuration: StreamGenerationConfiguration,
    partialStrings: StreamPartialStrings,
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
      let guideSelection = guideAttributes.first.map { attribute in
        Self.parseEdgeToolsGuide(in: attribute, context: context)
      }
      let ignoredAttribute = Self.ignoredAttribute(in: variableDecl)
      var isIgnored = ignoredAttribute != nil
      let hasDefaultValue = binding.initializer != nil
      let isOptional = type.streamIsOptional

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

      let field = StreamParseableField(
        name: identifierPattern.identifier,
        type: type,
        keys: [guideSelection?.key ?? propertyName],
        convertsKeys: guideSelection?.key == nil,
        initialCapacity: guideSelection?.initialCapacity,
        completedConversion: guideSelection?.completedConversion,
        partialStrings: guideSelection?.partialStrings
          ?? (guideSelection?.completedConversion == nil ? partialStrings : .streamString),
        defaultValue: binding.initializer?.value
      )
      return StoredProperty(
        streamField: field,
        keyExpression: Self.schemaKeyExpression(for: field, configuration: configuration),
        isIgnored: isIgnored,
        schemaFragments: guideSelection?.schemaFragments ?? []
      )
    }
  }

  private static func enumCases(
    in declaration: EnumDeclSyntax,
    configuration: StreamGenerationConfiguration,
    partialStrings: StreamPartialStrings
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
      let associatedValues = try parameters.enumerated()
        .map { index, parameter in
          let rawLabel = parameter.firstName?.text
          let label = rawLabel == "_" ? nil : rawLabel
          let schemaKey = label ?? "_\(index)"
          guard schemaKeys.insert(schemaKey).inserted else {
            throw MacroExpansionErrorMessage(
              "Enum case '\(name)' has multiple associated values represented by the key '\(schemaKey)'."
            )
          }
          let field = StreamParseableField(
            name: label == nil ? .wildcardToken() : parameter.firstName!,
            type: parameter.type,
            keys: [schemaKey],
            convertsKeys: label != nil,
            partialStrings: partialStrings
          )
          return AssociatedValue(
            streamField: field,
            keyExpression: Self.schemaKeyExpression(for: field, configuration: configuration),
            bindingName: "value\(index)"
          )
        }

      return EnumCase(
        sourceToken: element.name,
        keyExpression: Self.schemaKeyExpression(for: name, configuration: configuration),
        associatedValues: associatedValues
      )
    }
  }

  private static func schemaFragments(
    from attribute: AttributeSyntax
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
  ) -> EdgeToolsGuideSelection {
    var guide = EdgeToolsGuideSelection()
    guard case .argumentList(let arguments) = attribute.arguments else { return guide }

    for argument in arguments {
      do throws(MacroExpansionErrorMessage) {
        switch argument.label?.text {
        case "key":
          guard let key = Self.stringLiteralValue(from: argument.expression) else {
            throw MacroExpansionErrorMessage("key must be a string literal.")
          }
          guide.key = key
        case "initialCapacity" where !argument.expression.is(NilLiteralExprSyntax.self):
          guard let literal = argument.expression.as(IntegerLiteralExprSyntax.self),
            let capacity = Int(literal.literal.text.replacingOccurrences(of: "_", with: "")),
            capacity >= 0
          else {
            throw MacroExpansionErrorMessage(
              "initialCapacity must be a nonnegative integer literal."
            )
          }
          guide.initialCapacity = ExprSyntax("\(raw: String(capacity))")
        case "partialStrings" where !argument.expression.is(NilLiteralExprSyntax.self):
          guide.partialStrings = try Self.partialStrings(from: argument.expression)
        case "completedConversion":
          guard let member = argument.expression.as(MemberAccessExprSyntax.self),
            member.declName.baseName.text == "self", let base = member.base
          else {
            throw MacroExpansionErrorMessage(
              "completedConversion requires a strategy type followed by .self."
            )
          }
          guide.completedConversion = TypeSyntax("\(raw: base.trimmedDescription)")
        case nil:
          guide.schemaFragments.append(argument.expression.trimmedDescription)
        default:
          break
        }
      } catch {
        context.diagnose(
          Diagnostic(node: Syntax(argument), message: SimpleDiagnostic(error.message))
        )
      }
    }
    return guide
  }

  private static func generationSchemaProperty(
    from properties: [StoredProperty],
    modifierPrefix: String,
    schemaFragments: [String]
  ) -> DeclSyntax {
    let activeProperties = properties.filter { !$0.isIgnored }
    let propertyPairs =
      activeProperties.map { property in
        "\(property.keyExpression): \(property.schemaExpression)"
      }
      .joined(separator: ",\n          ")
    let requiredProperties = activeProperties.filter { !$0.isOptional }
      .map { property in
        property.keyExpression
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
      let propertyPairs = enumCase.associatedValues
        .map { value in
          "\(value.keyExpression): \(value.typeName).edgeToolsGenerationSchema"
        }
        .joined(separator: ",\n                      ")
      let required = enumCase.associatedValues.filter { !$0.isOptional }
        .map { $0.keyExpression }
        .joined(separator: ", ")
      if required.isEmpty {
        return """
          EdgeToolsGenerationSchema(
            .type(.object),
            .properties([
              \(enumCase.keyExpression): EdgeToolsGenerationSchema(
                .type(.object),
                .properties([
                  \(propertyPairs)
                ]),
                .additionalProperties(false)
              )
            ]),
            .required([\(enumCase.keyExpression)]),
            .additionalProperties(false)
          )
          """
      }
      return """
        EdgeToolsGenerationSchema(
          .type(.object),
          .properties([
            \(enumCase.keyExpression): EdgeToolsGenerationSchema(
              .type(.object),
              .properties([
                \(propertyPairs)
              ]),
              .required([\(required)]),
              .additionalProperties(false)
            )
          ]),
          .required([\(enumCase.keyExpression)]),
          .additionalProperties(false)
        )
        """
    }
    var fragments = [
      ".anyOf([\n            \(choices.joined(separator: ",\n            "))\n          ])"
    ]
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
    modifierPrefix: String,
    preservesStreamNull: Bool = false
  ) -> DeclSyntax {
    let assignments =
      properties.compactMap { property -> String? in
        if property.isIgnored {
          return property.hasDefaultValue ? nil : "self.\(property.name) = nil"
        }
        if preservesStreamNull {
          let type = property.streamField.type.streamUnwrappedOptionalType.trimmedDescription
          return "self.\(property.name) = try _edgeToolsPartialValue(object[\(property.keyExpression)], as: \(type).self)"
        }
        let value = "_edgeToolsValue(object, forKey: \(property.keyExpression))"
        if let conversion = property.streamField.completedConversion {
          let converted = "try \(conversion).value(edgeToolsValue: \(value))"
          let expression = property.isOptional ? "\(value) == .null ? nil : \(converted)" : converted
          return "self.\(property.name) = \(expression)"
        }
        return "self.\(property.name) = try \(property.initializerTypeName)(edgeToolsValue: \(value))"
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
    let caseInitializers =
      cases.map { enumCase in
        let keys = enumCase.associatedValues.filter { !$0.isOptional }
          .map { $0.keyExpression }
          .joined(separator: ", ")
        let arguments = enumCase.associatedValues
          .map { value in
            let expression =
              "try \(Self.initializerTypeName(for: value.typeName))(edgeToolsValue: _edgeToolsValue(payload, forKey: \(value.keyExpression)))"
            return value.sourceLabel.map { "\($0): \(expression)" } ?? expression
          }
          .joined(separator: ",\n          ")
        return """
          if let value = object[\(enumCase.keyExpression)] {
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
        let valueExpression: String
        if let conversion = property.streamField.completedConversion {
          valueExpression = property.isOptional
            ? "self.\(property.name).map { \(conversion).convertFromValue($0).edgeToolsValue }"
            : "\(conversion).convertFromValue(self.\(property.name)).edgeToolsValue"
        } else {
          valueExpression = property.isOptional
            ? "self.\(property.name)?.edgeToolsValue"
            : "self.\(property.name).edgeToolsValue"
        }
        return
          "(key: \(property.keyExpression), value: \(valueExpression))"
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
    let switchCases =
      cases.map { enumCase in
        let bindings = enumCase.associatedValues.map { "let \($0.bindingName)" }
          .joined(separator: ", ")
        let entries = enumCase.associatedValues
          .map { value in
            "(key: \(value.keyExpression), value: \(value.bindingName).edgeToolsValue)"
          }
          .joined(separator: ",\n            ")
        return """
          case .\(enumCase.sourceName)(\(bindings)):
            _edgeToolsBuildObjectValue(
              (key: \(enumCase.keyExpression), value: _edgeToolsBuildObjectValue(
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

// MARK: - Stream Parsing Synthesis

extension EdgeToolsGenerableMacro {
  private static func genericParameters(
    in declaration: some DeclGroupSyntax,
    context: some MacroExpansionContext
  ) -> [TokenSyntax] {
    let enclosing = context.lexicalContext.enumerated().compactMap { index, node -> Syntax? in
      if index == 0,
        node.asProtocol(NamedDeclSyntax.self)?.name.text
          == declaration.asProtocol(NamedDeclSyntax.self)?.name.text
      {
        return nil
      }
      return node
    }
    return ([Syntax(declaration)] + enclosing).flatMap { node -> [TokenSyntax] in
      guard node.isProtocol(DeclGroupSyntax.self) else {
        return []
      }
      return node.asProtocol(WithGenericParametersSyntax.self)?
        .genericParameterClause?.parameters.map(\.name.trimmed) ?? []
    }
  }

  private static func streamAccessModifier(
    for declaration: some DeclGroupSyntax,
    in context: some MacroExpansionContext
  ) -> String? {
    let isFileRestricted: (DeclModifierSyntax) -> Bool = {
      $0.name.tokenKind == .keyword(.private) || $0.name.tokenKind == .keyword(.fileprivate)
    }
    if declaration.modifiers.contains(where: isFileRestricted)
      || context.lexicalContext.contains(where: {
        $0.asProtocol(DeclGroupSyntax.self)?.modifiers.contains(where: isFileRestricted) ?? false
      })
    {
      return "fileprivate"
    }
    return Self.accessModifier(for: declaration)
  }

  private static func streamGenerationConfiguration(
    accessModifier: String?,
    genericParameters: [TokenSyntax] = [],
    from attribute: AttributeSyntax
  ) -> StreamGenerationConfiguration {
    let accessLevel: StreamGeneratedAccessLevel =
      switch accessModifier {
      case "public": .public
      case "package": .package
      case "fileprivate": .fileprivate
      default: .internal
      }
    return StreamGenerationConfiguration(
      viewMode: .unsafe,
      accessLevel: accessLevel,
      genericParameters: genericParameters,
      schemaCache: Self.argument(named: "schemaCache", in: attribute),
      keyDecodingStrategy: Self.argument(named: "keyDecodingStrategy", in: attribute)
    )
  }

  private static func streamDefaultCase(in declaration: EnumDeclSyntax) throws -> TokenSyntax? {
    var defaults = [TokenSyntax]()
    for member in declaration.memberBlock.members {
      guard let caseDecl = member.decl.as(EnumCaseDeclSyntax.self) else { continue }
      let isDefault = caseDecl.attributes.contains { element in
        guard let attribute = element.as(AttributeSyntax.self) else { return false }
        return ["StreamParseableDefault", "StreamParsing.StreamParseableDefault"]
          .contains(attribute.attributeName.trimmedDescription)
      }
      guard isDefault else { continue }
      guard caseDecl.elements.count == 1, let name = caseDecl.elements.first?.name else {
        throw MacroExpansionErrorMessage(
          "@StreamParseableDefault must mark a declaration with one enum case."
        )
      }
      defaults.append(name)
    }
    guard !defaults.isEmpty else { return nil }
    guard defaults.count == 1 else {
      throw MacroExpansionErrorMessage(
        "Stream parsing synthesis for an enum requires exactly one @StreamParseableDefault case."
      )
    }
    return defaults[0]
  }

  private static func generablePartialCustomization(
    fields: [StreamPartialFieldDescriptor],
    keyExpressions: [String],
    schemaFragments: [[String]] = [],
    preservesStreamNull: Bool = false,
    accessModifier: String?
  ) -> StreamPartialCustomization {
    let generatedProperties = zip(fields, keyExpressions).enumerated()
      .map { index, pair in
        let (field, keyExpression) = pair
        return StoredProperty(
          streamField: StreamParseableField(
            name: field.memberName,
            type: field.storageType,
            keys: field.keys
          ),
          keyExpression: keyExpression,
          schemaFragments: schemaFragments.isEmpty ? [] : schemaFragments[index]
        )
      }
    let modifierPrefix = Self.modifierPrefix(for: accessModifier)
    return StreamPartialCustomization(
      conformances: [TypeSyntax("EdgeToolsGenerable")],
      members: MemberBlockItemListSyntax([
        MemberBlockItemSyntax(
          decl: Self.generationSchemaProperty(
            from: generatedProperties,
            modifierPrefix: modifierPrefix,
            schemaFragments: []
          )
        ),
        MemberBlockItemSyntax(
          decl: Self.valueInitializer(
            from: generatedProperties,
            modifierPrefix: modifierPrefix,
            preservesStreamNull: preservesStreamNull
          )
        ),
        MemberBlockItemSyntax(
          decl: Self.valueProperty(
            from: generatedProperties,
            modifierPrefix: modifierPrefix
          )
        )
      ])
    )
  }

}

// MARK: - Streaming Options

extension EdgeToolsGenerableMacro {
  private static func argument(named name: String, in attribute: AttributeSyntax) -> ExprSyntax? {
    guard case .argumentList(let arguments) = attribute.arguments else { return nil }
    return arguments.first { $0.label?.text == name }?.expression
  }

  private static func partialStrings(
    from expression: ExprSyntax
  ) throws(MacroExpansionErrorMessage) -> StreamPartialStrings {
    guard let member = expression.as(MemberAccessExprSyntax.self),
      member.base.map({
        ["PartialStringStorage", "StreamParsing.PartialStringStorage"]
          .contains($0.trimmedDescription)
      }) ?? true
    else {
      throw MacroExpansionErrorMessage("partialStrings requires .streamString or .string.")
    }
    switch member.declName.baseName.text {
    case "streamString": return .streamString
    case "string": return .string
    default:
      throw MacroExpansionErrorMessage("partialStrings requires .streamString or .string.")
    }
  }

  private static func schemaKeyExpression(
    for field: StreamParseableField,
    configuration: StreamGenerationConfiguration
  ) -> String {
    let key = field.keys[0]
    return field.convertsKeys
      ? Self.schemaKeyExpression(for: key, configuration: configuration)
      : Self.quotedStringLiteral(key)
  }

  private static func schemaKeyExpression(
    for key: String,
    configuration: StreamGenerationConfiguration
  ) -> String {
    if let decoded = configuration.decodedKey(for: key) {
      return Self.quotedStringLiteral(decoded)
    }
    return
      "(\(configuration.keyDecodingStrategy!.trimmedDescription) as StreamParsing.StreamKeyDecodingStrategy).key(for: \(Self.quotedStringLiteral(key)))"
  }
}
