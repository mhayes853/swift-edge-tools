import Foundation
import SwiftDiagnostics
import SwiftSyntax
import SwiftSyntaxBuilder
import SwiftSyntaxMacros

public enum NeedleGenerableMacro: ExtensionMacro, MemberMacro, MemberAttributeMacro {
  public static func expansion(
    of node: AttributeSyntax,
    providingMembersOf declaration: some DeclGroupSyntax,
    conformingTo protocols: [TypeSyntax],
    in context: some MacroExpansionContext
  ) throws -> [DeclSyntax] {
    let structDecl = try Self.requireStructDecl(declaration: declaration)
    let properties = Self.storedProperties(in: structDecl, context: context)
    let schemaMetadata = Self.schemaMetadata(from: node)
    let accessModifier = Self.accessModifier(for: structDecl)
    let hasNeedleGenerationSchema = Self.hasExistingNeedleGenerationSchema(in: structDecl)
    let hasNeedleValueInitializer = Self.hasExistingNeedleValueInitializer(in: structDecl)
    let modifierPrefix = Self.modifierPrefix(for: accessModifier)

    var members = [DeclSyntax]()

    if !hasNeedleGenerationSchema {
      members.append(
        Self.needleGenerationSchemaProperty(
          from: properties,
          modifierPrefix: modifierPrefix,
          schemaMetadata: schemaMetadata
        )
      )
    }

    if !hasNeedleValueInitializer {
      members.append(
        Self.needleValueInitializer(
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
        extension \(raw: typeName): NeedleGenerable {}
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
    guard declaration.as(StructDeclSyntax.self) != nil else { return [] }
    guard let variableDecl = member.as(VariableDeclSyntax.self) else { return [] }
    guard !Self.isStatic(variableDecl) else { return [] }
    guard variableDecl.bindings.allSatisfy({ !Self.isComputedProperty($0) }) else {
      return []
    }

    return []
  }

  private static func requireStructDecl(
    declaration: some DeclGroupSyntax
  ) throws -> StructDeclSyntax {
    guard let structDecl = declaration.as(StructDeclSyntax.self) else {
      throw MacroExpansionErrorMessage(
        "@NeedleGenerable can only be applied to struct declarations."
      )
    }
    return structDecl
  }

  private static func isStatic(_ variableDecl: VariableDeclSyntax) -> Bool {
    variableDecl.modifiers.contains { $0.name.tokenKind == .keyword(.static) }
  }

  private static func hasExistingNeedleGenerationSchema(in declaration: StructDeclSyntax) -> Bool {
    declaration.memberBlock.members.contains { member in
      guard let variableDecl = member.decl.as(VariableDeclSyntax.self),
        Self.isStatic(variableDecl)
      else {
        return false
      }

      return variableDecl.bindings.contains { binding in
        guard let identifierPattern = binding.pattern.as(IdentifierPatternSyntax.self) else {
          return false
        }
        return identifierPattern.identifier.text == "needleGenerationSchema"
      }
    }
  }

  private static func hasExistingNeedleValueInitializer(in declaration: StructDeclSyntax) -> Bool {
    declaration.memberBlock.members.contains { member in
      guard let initializer = member.decl.as(InitializerDeclSyntax.self) else { return false }
      let parameters = initializer.signature.parameterClause.parameters
      guard parameters.count == 1, let parameter = parameters.first else { return false }
      return parameter.firstName.text == "needleValue"
    }
  }

  private static func needleGenerationSchemaProperty(
    from properties: [StoredProperty],
    modifierPrefix: String,
    schemaMetadata: SchemaMetadata
  ) -> DeclSyntax {
    let activeProperties = properties.filter { !$0.isIgnored }

    let propertyPairs =
      activeProperties
      .map { property in
        "\"\(property.schemaKey)\": \(property.schemaExpression ?? "\(property.typeName).needleGenerationSchema")"
      }
      .joined(separator: ",\n          ")

    let requiredProperties =
      activeProperties
      .filter { !Self.isOptionalTypeName($0.typeName) }
      .map { property in
        "\"\(property.schemaKey)\""
      }
      .joined(separator: ", ")

    let objectMetadataArguments = Self.objectMetadataArguments(schemaMetadata)

    if activeProperties.isEmpty {
      if objectMetadataArguments.isEmpty {
        return """
          \(raw: modifierPrefix)static var needleGenerationSchema: NeedleGenerationSchema {
            .object(valueSchema: .object())
          }
          """
      }

      return """
        \(raw: modifierPrefix)static var needleGenerationSchema: NeedleGenerationSchema {
          .object(\(raw: objectMetadataArguments), valueSchema: .object())
        }
        """
    }

    if requiredProperties.isEmpty {
      if objectMetadataArguments.isEmpty {
        return """
          \(raw: modifierPrefix)static var needleGenerationSchema: NeedleGenerationSchema {
            .object(
              valueSchema: .object(
                properties: [
                  \(raw: propertyPairs)
                ]
              )
            )
          }
          """
      }

      return """
        \(raw: modifierPrefix)static var needleGenerationSchema: NeedleGenerationSchema {
          .object(
            \(raw: objectMetadataArguments),
            valueSchema: .object(
              properties: [
                \(raw: propertyPairs)
              ]
            )
          )
        }
        """
    }

    if objectMetadataArguments.isEmpty {
      return """
        \(raw: modifierPrefix)static var needleGenerationSchema: NeedleGenerationSchema {
          .object(
            valueSchema: .object(
              properties: [
                \(raw: propertyPairs)
              ],
              required: [\(raw: requiredProperties)]
            )
          )
        }
        """
    }

    return """
      \(raw: modifierPrefix)static var needleGenerationSchema: NeedleGenerationSchema {
        .object(
          \(raw: objectMetadataArguments),
          valueSchema: .object(
            properties: [
              \(raw: propertyPairs)
            ],
            required: [\(raw: requiredProperties)]
          )
        )
      }
      """
  }

  private static func needleValueInitializer(
    from properties: [StoredProperty],
    modifierPrefix: String
  ) -> DeclSyntax {
    let assignments =
      properties
      .compactMap { property -> String? in
        if property.isIgnored {
          if property.hasDefaultValue {
            return nil
          }
          return "self.\(property.name) = nil"
        }

        return
          "self.\(property.name) = try \(property.initializerTypeName)(needleValue: _needleValue(object, forKey: \(Self.quotedStringLiteral(property.schemaKey))))"
      }
      .joined(separator: "\n")

    if assignments.isEmpty {
      return """
        \(raw: modifierPrefix)init(needleValue: NeedleValue) throws {
          _ = try _needleRequireObjectValue(needleValue)
        }
        """
    }

    return """
      \(raw: modifierPrefix)init(needleValue: NeedleValue) throws {
        let object = try _needleRequireObjectValue(needleValue)
        \(raw: assignments)
      }
      """
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

  private static func isOptionalTypeName(_ typeName: String) -> Bool {
    typeName.hasSuffix("?") || typeName.hasPrefix("Optional<")
  }
}

// MARK: - StoredProperty

extension NeedleGenerableMacro {
  private enum SemanticSchemaKind {
    case string
    case number
    case integer
    case boolean
  }

  private static let stringSemanticTypes = Set(["String"])

  private static let numberSemanticTypes = Set(["Double", "Float", "CGFloat", "Decimal"])

  private static let integerSemanticTypes = Set([
    "Int",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Int128",
    "UInt128"
  ])

  private static let booleanSemanticTypes = Set(["Bool"])

  private struct SchemaMetadata {
    let title: String?
    let description: String?
  }

  private struct ParsedTypeName {
    let base: String
    let isOptional: Bool
    let isArray: Bool
    let arrayElementTypeName: String?
    let arrayElementBaseType: String?
    let isDictionary: Bool
    let dictionaryKeyTypeName: String?
    let dictionaryKeyBaseType: String?
    let dictionaryValueTypeName: String?
    let dictionaryValueBaseType: String?
  }

  private struct StoredProperty {
    let name: String
    let schemaKey: String
    let typeName: String
    let initializerTypeName: String
    let isIgnored: Bool
    let isOptional: Bool
    let hasDefaultValue: Bool
    let schemaExpression: String?
  }

  private indirect enum SchemaSpecifier {
    case inferred
    case string(String?)
    case number(String?)
    case integer(String?)
    case boolean
    case array(arguments: String?, itemSchema: SchemaSpecifier?)
    case object(arguments: String?, additionalPropertiesSchema: SchemaSpecifier?)
    case custom(String)
  }

  private struct NeedleGuideSelection {
    let attribute: AttributeSyntax
    let key: String?
    let description: String?
    let schemaSpecifier: SchemaSpecifier
  }

  private static func schemaMetadata(from node: AttributeSyntax) -> SchemaMetadata {
    guard case .argumentList(let arguments) = node.arguments else {
      return SchemaMetadata(title: nil, description: nil)
    }

    var title: String?
    var description: String?

    for argument in arguments {
      guard let label = argument.label?.text else { continue }
      switch label {
      case "title":
        title = Self.stringLiteralValue(from: argument.expression)
      case "description":
        description = Self.stringLiteralValue(from: argument.expression)
      default:
        continue
      }
    }

    return SchemaMetadata(title: title, description: description)
  }

  private static func objectMetadataArguments(_ metadata: SchemaMetadata) -> String {
    var arguments = [String]()
    if let title = metadata.title {
      arguments.append("title: \(Self.quotedStringLiteral(title))")
    }
    if let description = metadata.description {
      arguments.append("description: \(Self.quotedStringLiteral(description))")
    }
    return arguments.joined(separator: ", ")
  }

  private static func storedProperties(
    in declaration: StructDeclSyntax,
    context: some MacroExpansionContext
  ) -> [StoredProperty] {
    declaration.memberBlock.members.reduce(into: [StoredProperty]()) { properties, member in
      guard let variableDecl = member.decl.as(VariableDeclSyntax.self) else {
        return
      }
      properties.append(contentsOf: Self.storedProperties(from: variableDecl, context: context))
    }
  }

  private static func storedProperties(
    from variableDecl: VariableDeclSyntax,
    context: some MacroExpansionContext
  ) -> [StoredProperty] {
    if Self.isStatic(variableDecl) {
      Self.diagnoseUnsupportedNeedleGenerationSchemaMember(in: variableDecl, context: context)
      return []
    }

    return variableDecl.bindings.compactMap { binding -> StoredProperty? in
      if Self.isComputedProperty(binding) {
        Self.diagnoseUnsupportedNeedleGenerationSchemaMember(in: variableDecl, context: context)
        return nil
      }

      guard let identifierPattern = binding.pattern.as(IdentifierPatternSyntax.self) else {
        return nil
      }

      guard let type = binding.typeAnnotation?.type else {
        Self.diagnoseMissingTypeAnnotation(in: binding, context: context)
        return nil
      }

      return Self.storedProperty(
        from: variableDecl,
        propertyName: identifierPattern.identifier.text,
        type: type,
        context: context
      )
    }
  }

  private static func storedProperty(
    from variableDecl: VariableDeclSyntax,
    propertyName: String,
    type: TypeSyntax,
    context: some MacroExpansionContext
  ) -> StoredProperty {
    let ignoredAttribute = Self.needleGenerationSchemaIgnoredAttribute(in: variableDecl)
    let isIgnored = ignoredAttribute != nil
    let needleGenerationSchemaPropertyAttribute = Self.needleGenerationSchemaPropertyAttribute(
      in: variableDecl,
      propertyName: propertyName,
      context: context
    )
    let propertySelection = needleGenerationSchemaPropertyAttribute.flatMap {
      Self.parseNeedleGuide(
        in: $0,
        propertyName: propertyName,
        context: context
      )
    }
    let schemaKey = propertySelection?.key ?? propertyName
    let typeName = type.trimmedDescription
    let parsedType = Self.parseTypeName(typeName)
    let hasDefaultValue = variableDecl.bindings.contains { binding in
      guard let identifierPattern = binding.pattern.as(IdentifierPatternSyntax.self) else {
        return false
      }
      return identifierPattern.identifier.text == propertyName && binding.initializer != nil
    }

    if isIgnored && needleGenerationSchemaPropertyAttribute != nil {
      if let conflictingAttribute = needleGenerationSchemaPropertyAttribute {
        Self.diagnoseConflictingSchemaAttributes(
          in: conflictingAttribute,
          propertyName: propertyName,
          context: context
        )
      }
    }

    if isIgnored && needleGenerationSchemaPropertyAttribute == nil && !parsedType.isOptional
      && !hasDefaultValue
    {
      if let ignoredAttribute {
        Self.diagnoseInvalidNeedleIgnoredAttribute(in: ignoredAttribute, context: context)
      }
    }

    let overriddenSchemaExpression = Self.schemaExpression(
      for: propertySelection,
      propertyName: propertyName,
      typeName: typeName,
      context: context
    )
    let schemaExpression = Self.schemaExpressionWithDescription(
      schemaExpression: overriddenSchemaExpression,
      typeName: typeName,
      description: propertySelection?.description
    )

    return StoredProperty(
      name: propertyName,
      schemaKey: schemaKey,
      typeName: typeName,
      initializerTypeName: Self.initializerTypeName(for: typeName),
      isIgnored: isIgnored,
      isOptional: parsedType.isOptional,
      hasDefaultValue: hasDefaultValue,
      schemaExpression: schemaExpression
    )
  }

  private static func schemaExpression(
    for propertySelection: NeedleGuideSelection?,
    propertyName: String,
    typeName: String,
    context: some MacroExpansionContext
  ) -> String? {
    guard let propertySelection else { return nil }
    let parsedType = Self.parseTypeName(typeName)

    switch propertySelection.schemaSpecifier {
    case .inferred:
      return nil
    case .string(let arguments):
      guard Self.isValid(type: parsedType.base, for: .string), !parsedType.isArray,
        !parsedType.isDictionary
      else {
        Self.diagnoseInvalidSemanticSchemaType(
          in: propertySelection.attribute,
          propertyName: propertyName,
          typeName: typeName,
          expected: "String",
          kind: ".string",
          context: context
        )
        return nil
      }
      return Self.semanticSchemaExpression(
        for: .string,
        arguments: arguments,
        isOptional: parsedType.isOptional
      )
    case .number(let arguments):
      guard Self.isValid(type: parsedType.base, for: .number), !parsedType.isArray,
        !parsedType.isDictionary
      else {
        Self.diagnoseInvalidSemanticSchemaType(
          in: propertySelection.attribute,
          propertyName: propertyName,
          typeName: typeName,
          expected: "number (Double, Float, CGFloat, Decimal)",
          kind: ".number",
          context: context
        )
        return nil
      }
      return Self.semanticSchemaExpression(
        for: .number,
        arguments: arguments,
        isOptional: parsedType.isOptional
      )
    case .integer(let arguments):
      guard Self.isValid(type: parsedType.base, for: .integer), !parsedType.isArray,
        !parsedType.isDictionary
      else {
        Self.diagnoseInvalidSemanticSchemaType(
          in: propertySelection.attribute,
          propertyName: propertyName,
          typeName: typeName,
          expected:
            "integer (Int, Int8, Int16, Int32, Int64, UInt, UInt8, UInt16, UInt32, UInt64, Int128, UInt128)",
          kind: ".integer",
          context: context
        )
        return nil
      }
      return Self.semanticSchemaExpression(
        for: .integer,
        arguments: arguments,
        isOptional: parsedType.isOptional
      )
    case .boolean:
      guard Self.isValid(type: parsedType.base, for: .boolean), !parsedType.isArray,
        !parsedType.isDictionary
      else {
        Self.diagnoseInvalidSemanticSchemaType(
          in: propertySelection.attribute,
          propertyName: propertyName,
          typeName: typeName,
          expected: "Bool",
          kind: ".boolean",
          context: context
        )
        return nil
      }
      return Self.semanticSchemaExpression(
        for: .boolean,
        arguments: nil,
        isOptional: parsedType.isOptional
      )
    case .array(let arguments, let itemSchema):
      guard parsedType.isArray, let elementTypeName = parsedType.arrayElementTypeName else {
        Self.diagnoseInvalidContainerSchemaType(
          in: propertySelection.attribute,
          propertyName: propertyName,
          typeName: typeName,
          expected: "array",
          kind: ".array",
          context: context
        )
        return nil
      }
      if let itemSchema {
        guard
          Self.validateSchemaSpecifier(
            itemSchema,
            against: elementTypeName,
            propertyName: propertyName,
            attribute: propertySelection.attribute,
            context: context
          )
        else {
          return nil
        }
      }
      let arrayArguments = Self.arraySchemaArguments(
        rawArguments: arguments,
        inferredElementTypeName: elementTypeName
      )
      return parsedType.isOptional
        ? ".union(array: .array(\(arrayArguments)), null: true)"
        : ".array(\(arrayArguments))"
    case .object(let arguments, let additionalPropertiesSchema):
      guard parsedType.isDictionary else {
        Self.diagnoseInvalidContainerSchemaType(
          in: propertySelection.attribute,
          propertyName: propertyName,
          typeName: typeName,
          expected: "dictionary",
          kind: ".object",
          context: context
        )
        return nil
      }
      guard parsedType.dictionaryKeyBaseType == "String" else {
        Self.diagnoseDictionarySchemaRequiresStringKeys(
          in: propertySelection.attribute,
          propertyName: propertyName,
          typeName: typeName,
          context: context
        )
        return nil
      }
      if let additionalPropertiesSchema, let valueTypeName = parsedType.dictionaryValueTypeName {
        guard
          Self.validateSchemaSpecifier(
            additionalPropertiesSchema,
            against: valueTypeName,
            propertyName: propertyName,
            attribute: propertySelection.attribute,
            context: context
          )
        else {
          return nil
        }
      }
      let objectArguments = Self.objectSchemaArguments(
        rawArguments: arguments,
        inferredValueTypeName: parsedType.dictionaryValueTypeName
      )
      return parsedType.isOptional
        ? ".union(object: .object(\(objectArguments)), null: true)"
        : ".object(\(objectArguments))"
    case .custom(let expression):
      return expression
    }
  }

  private static func validateSchemaSpecifier(
    _ schemaSpecifier: SchemaSpecifier,
    against typeName: String,
    propertyName: String,
    attribute: AttributeSyntax,
    context: some MacroExpansionContext
  ) -> Bool {
    let parsedType = Self.parseTypeName(typeName)

    switch schemaSpecifier {
    case .inferred, .custom:
      return true
    case .string:
      guard Self.isValid(type: parsedType.base, for: .string), !parsedType.isArray,
        !parsedType.isDictionary
      else {
        Self.diagnoseInvalidSemanticSchemaType(
          in: attribute,
          propertyName: propertyName,
          typeName: typeName,
          expected: "String",
          kind: ".string",
          context: context
        )
        return false
      }
      return true
    case .number:
      guard Self.isValid(type: parsedType.base, for: .number), !parsedType.isArray,
        !parsedType.isDictionary
      else {
        Self.diagnoseInvalidSemanticSchemaType(
          in: attribute,
          propertyName: propertyName,
          typeName: typeName,
          expected: "number (Double, Float, CGFloat, Decimal)",
          kind: ".number",
          context: context
        )
        return false
      }
      return true
    case .integer:
      guard Self.isValid(type: parsedType.base, for: .integer), !parsedType.isArray,
        !parsedType.isDictionary
      else {
        Self.diagnoseInvalidSemanticSchemaType(
          in: attribute,
          propertyName: propertyName,
          typeName: typeName,
          expected:
            "integer (Int, Int8, Int16, Int32, Int64, UInt, UInt8, UInt16, UInt32, UInt64, Int128, UInt128)",
          kind: ".integer",
          context: context
        )
        return false
      }
      return true
    case .boolean:
      guard Self.isValid(type: parsedType.base, for: .boolean), !parsedType.isArray,
        !parsedType.isDictionary
      else {
        Self.diagnoseInvalidSemanticSchemaType(
          in: attribute,
          propertyName: propertyName,
          typeName: typeName,
          expected: "Bool",
          kind: ".boolean",
          context: context
        )
        return false
      }
      return true
    case .array(_, let itemSchema):
      guard parsedType.isArray, let elementTypeName = parsedType.arrayElementTypeName else {
        Self.diagnoseInvalidContainerSchemaType(
          in: attribute,
          propertyName: propertyName,
          typeName: typeName,
          expected: "array",
          kind: ".array",
          context: context
        )
        return false
      }
      guard let itemSchema else { return true }
      return Self.validateSchemaSpecifier(
        itemSchema,
        against: elementTypeName,
        propertyName: propertyName,
        attribute: attribute,
        context: context
      )
    case .object(_, let additionalPropertiesSchema):
      guard parsedType.isDictionary else {
        Self.diagnoseInvalidContainerSchemaType(
          in: attribute,
          propertyName: propertyName,
          typeName: typeName,
          expected: "dictionary",
          kind: ".object",
          context: context
        )
        return false
      }
      guard parsedType.dictionaryKeyBaseType == "String" else {
        Self.diagnoseDictionarySchemaRequiresStringKeys(
          in: attribute,
          propertyName: propertyName,
          typeName: typeName,
          context: context
        )
        return false
      }
      guard let additionalPropertiesSchema, let valueTypeName = parsedType.dictionaryValueTypeName
      else {
        return true
      }
      return Self.validateSchemaSpecifier(
        additionalPropertiesSchema,
        against: valueTypeName,
        propertyName: propertyName,
        attribute: attribute,
        context: context
      )
    }
  }

  private static func objectSchemaArguments(
    rawArguments: String?,
    inferredValueTypeName: String?
  ) -> String {
    let normalizedArguments = rawArguments?
      .replacingOccurrences(
        of: "\\s+",
        with: "",
        options: .regularExpression
      )
    let hasAdditionalPropertiesArgument =
      normalizedArguments?.contains("additionalProperties:") ?? false

    let resolvedValueSchemaExpression =
      "\(inferredValueTypeName ?? "Any").needleGenerationSchema"

    if let rawArguments, !rawArguments.isEmpty {
      if hasAdditionalPropertiesArgument {
        return rawArguments
      } else {
        return "\(rawArguments), additionalProperties: \(resolvedValueSchemaExpression)"
      }
    }

    return "additionalProperties: \(resolvedValueSchemaExpression)"
  }

  private static func arraySchemaArguments(
    rawArguments: String?,
    inferredElementTypeName: String
  ) -> String {
    let normalizedArguments = rawArguments?
      .replacingOccurrences(
        of: "\\s+",
        with: "",
        options: .regularExpression
      )
    let hasItemsArgument =
      normalizedArguments?.contains(",items:") ?? false
      || normalizedArguments?.hasPrefix("items:") ?? false
    let hasPrefixItemsArgument =
      normalizedArguments?.contains(",prefixItems:") ?? false
      || normalizedArguments?.hasPrefix("prefixItems:") ?? false

    let resolvedItemSchemaExpression = "\(inferredElementTypeName).needleGenerationSchema"
    let itemsArgument = "items: \(resolvedItemSchemaExpression)"

    if let rawArguments, !rawArguments.isEmpty {
      if hasItemsArgument || hasPrefixItemsArgument {
        return rawArguments
      } else {
        return "\(itemsArgument), \(rawArguments)"
      }
    }

    return itemsArgument
  }

  private static func schemaExpressionWithDescription(
    schemaExpression: String?,
    typeName: String,
    description: String?
  ) -> String? {
    guard let description else { return schemaExpression }
    let expression = schemaExpression ?? "\(typeName).needleGenerationSchema"
    return
      "_needleMergeGenerationSchema(\(expression), description: \(Self.quotedStringLiteral(description)))"
  }

  private static func parseTypeName(_ typeName: String) -> ParsedTypeName {
    let trimmed = typeName.replacingOccurrences(of: " ", with: "")
    let unwrappedType: String
    let isOptional: Bool

    if trimmed.hasSuffix("?") {
      unwrappedType = String(trimmed.dropLast())
      isOptional = true
    } else if let innerType = Self.optionalInnerType(in: trimmed) {
      unwrappedType = innerType
      isOptional = true
    } else {
      unwrappedType = trimmed
      isOptional = false
    }

    if let (keyType, valueType) = Self.dictionaryElementTypes(in: unwrappedType) {
      return ParsedTypeName(
        base: Self.baseTypeName(for: unwrappedType),
        isOptional: isOptional,
        isArray: false,
        arrayElementTypeName: nil,
        arrayElementBaseType: nil,
        isDictionary: true,
        dictionaryKeyTypeName: keyType,
        dictionaryKeyBaseType: Self.baseTypeName(for: Self.unwrappedTypeName(keyType)),
        dictionaryValueTypeName: valueType,
        dictionaryValueBaseType: Self.baseTypeName(for: Self.unwrappedTypeName(valueType))
      )
    }

    if let arrayElementType = Self.arrayElementType(in: unwrappedType) {
      return ParsedTypeName(
        base: Self.baseTypeName(for: unwrappedType),
        isOptional: isOptional,
        isArray: true,
        arrayElementTypeName: arrayElementType,
        arrayElementBaseType: Self.baseTypeName(for: Self.unwrappedTypeName(arrayElementType)),
        isDictionary: false,
        dictionaryKeyTypeName: nil,
        dictionaryKeyBaseType: nil,
        dictionaryValueTypeName: nil,
        dictionaryValueBaseType: nil
      )
    }

    return ParsedTypeName(
      base: Self.baseTypeName(for: unwrappedType),
      isOptional: isOptional,
      isArray: false,
      arrayElementTypeName: nil,
      arrayElementBaseType: nil,
      isDictionary: false,
      dictionaryKeyTypeName: nil,
      dictionaryKeyBaseType: nil,
      dictionaryValueTypeName: nil,
      dictionaryValueBaseType: nil
    )
  }

  private static func unwrappedTypeName(_ typeName: String) -> String {
    if typeName.hasSuffix("?") {
      return String(typeName.dropLast())
    }
    return Self.optionalInnerType(in: typeName) ?? typeName
  }

  private static func initializerTypeName(for typeName: String) -> String {
    let trimmed = typeName.replacingOccurrences(of: " ", with: "")
    if trimmed.hasSuffix("?") {
      return "Optional<\(String(trimmed.dropLast()))>"
    }
    return trimmed
  }

  private static func optionalInnerType(in typeName: String) -> String? {
    let optionalPrefixes = ["Optional<", "Swift.Optional<"]
    guard let prefix = optionalPrefixes.first(where: { typeName.hasPrefix($0) }),
      typeName.hasSuffix(">")
    else {
      return nil
    }
    return String(typeName.dropFirst(prefix.count).dropLast())
  }

  private static func baseTypeName(for typeName: String) -> String {
    typeName.split(separator: ".").last.map(String.init) ?? typeName
  }

  private static func arrayElementType(in typeName: String) -> String? {
    if typeName.hasPrefix("[") && typeName.hasSuffix("]") {
      return String(typeName.dropFirst().dropLast())
    }

    let prefixes = ["Array<", "Swift.Array<"]
    guard let prefix = prefixes.first(where: { typeName.hasPrefix($0) }), typeName.hasSuffix(">")
    else {
      return nil
    }
    return String(typeName.dropFirst(prefix.count).dropLast())
  }

  private static func dictionaryElementTypes(in typeName: String) -> (String, String)? {
    let prefixes = ["Dictionary<", "Swift.Dictionary<", "["]
    guard let prefix = prefixes.first(where: { typeName.hasPrefix($0) })
    else {
      return nil
    }

    if prefix == "[" && typeName.contains(":") {
      let inner = String(typeName.dropFirst().dropLast())
      let parts = inner.split(separator: ":", maxSplits: 1).map(String.init)
      guard parts.count == 2 else { return nil }
      return (
        parts[0].trimmingCharacters(in: .whitespaces), parts[1].trimmingCharacters(in: .whitespaces)
      )
    }

    if prefix == "Dictionary<" || prefix == "Swift.Dictionary<" {
      let inner = String(typeName.dropFirst(prefix.count).dropLast())
      let parts = inner.split(separator: ",", maxSplits: 1).map(String.init)
      guard parts.count == 2 else { return nil }
      return (
        parts[0].trimmingCharacters(in: .whitespaces), parts[1].trimmingCharacters(in: .whitespaces)
      )
    }

    return nil
  }

  private static func parseNeedleGuide(
    in attribute: AttributeSyntax,
    propertyName: String,
    context: some MacroExpansionContext
  ) -> NeedleGuideSelection? {
    guard case .argumentList(let arguments) = attribute.arguments else {
      return NeedleGuideSelection(
        attribute: attribute,
        key: nil,
        description: nil,
        schemaSpecifier: .inferred
      )
    }

    var key: String?
    var description: String?
    var schemaSpecifier: SchemaSpecifier = .inferred

    for argument in arguments {
      if let label = argument.label?.text {
        switch label {
        case "key":
          guard let value = Self.stringLiteralValue(from: argument.expression) else {
            Self.diagnoseInvalidNeedleGuideAttribute(
              in: attribute,
              propertyName: propertyName,
              context: context
            )
            return nil
          }
          key = value
        case "description":
          guard let value = Self.stringLiteralValue(from: argument.expression) else {
            Self.diagnoseInvalidNeedleGuideAttribute(
              in: attribute,
              propertyName: propertyName,
              context: context
            )
            return nil
          }
          description = value
        default:
          Self.diagnoseInvalidNeedleGuideAttribute(
            in: attribute,
            propertyName: propertyName,
            context: context
          )
          return nil
        }
      } else {
        guard let parsedSchemaSpecifier = Self.parseSchemaSpecifier(from: argument.expression)
        else {
          Self.diagnoseInvalidNeedleGuideAttribute(
            in: attribute,
            propertyName: propertyName,
            context: context
          )
          return nil
        }
        schemaSpecifier = parsedSchemaSpecifier
      }
    }

    return NeedleGuideSelection(
      attribute: attribute,
      key: key,
      description: description,
      schemaSpecifier: schemaSpecifier
    )
  }

  private static func parseSchemaSpecifier(from expression: ExprSyntax) -> SchemaSpecifier? {
    let text = expression.trimmedDescription

    guard !text.isEmpty else { return nil }

    if text == ".inferred" || text == "_NeedleGuideSchema.inferred" {
      return .inferred
    }

    if text == ".boolean" || text == "_NeedleGuideSchema.boolean" || text == ".boolean()"
      || text == "_NeedleGuideSchema.boolean()"
    {
      return .boolean
    }

    guard let openParen = text.firstIndex(of: "("), text.hasSuffix(")") else {
      return nil
    }

    let prefix = String(text[..<openParen])
    let suffixName = prefix.split(separator: ".").last.map(String.init) ?? prefix
    let argumentsStart = text.index(after: openParen)
    let argumentsEnd = text.index(before: text.endIndex)
    let arguments = String(text[argumentsStart..<argumentsEnd])
      .trimmingCharacters(in: .whitespacesAndNewlines)

    switch suffixName {
    case "string":
      if let functionCall = expression.as(FunctionCallExprSyntax.self) {
        let serializedArguments = Self.serializedSchemaArguments(
          from: functionCall.arguments,
          convertRegexPatternLiteralToString: true
        )
        return .string(serializedArguments)
      }
      return .string(arguments.isEmpty ? nil : arguments)
    case "number":
      return .number(arguments.isEmpty ? nil : arguments)
    case "integer":
      return .integer(arguments.isEmpty ? nil : arguments)
    case "array":
      if let functionCall = expression.as(FunctionCallExprSyntax.self) {
        let serializedArguments = Self.serializedSchemaArguments(
          from: functionCall.arguments,
          convertRegexPatternLiteralToString: false
        )
        let itemSchema = Self.arrayItemSchemaSpecifier(from: functionCall.arguments)
        return .array(arguments: serializedArguments, itemSchema: itemSchema)
      }
      return .array(arguments: arguments.isEmpty ? nil : arguments, itemSchema: nil)
    case "object":
      if let functionCall = expression.as(FunctionCallExprSyntax.self) {
        let serializedArguments = Self.serializedSchemaArguments(
          from: functionCall.arguments,
          convertRegexPatternLiteralToString: false
        )
        let additionalPropertiesSchema = Self.objectAdditionalPropertiesSchemaSpecifier(
          from: functionCall.arguments
        )
        return .object(
          arguments: serializedArguments,
          additionalPropertiesSchema: additionalPropertiesSchema
        )
      }
      return .object(
        arguments: arguments.isEmpty ? nil : arguments,
        additionalPropertiesSchema: nil
      )
    case "custom":
      guard !arguments.isEmpty else { return nil }
      return .custom(arguments)
    case "inferred":
      return .inferred
    case "boolean":
      return .boolean
    default:
      return nil
    }
  }

  private static func arrayItemSchemaSpecifier(
    from arguments: LabeledExprListSyntax
  ) -> SchemaSpecifier? {
    guard let itemsArgument = arguments.first(where: { $0.label?.text == "items" }) else {
      return nil
    }
    return Self.parseSchemaSpecifier(from: itemsArgument.expression)
  }

  private static func objectAdditionalPropertiesSchemaSpecifier(
    from arguments: LabeledExprListSyntax
  ) -> SchemaSpecifier? {
    guard
      let additionalPropertiesArgument = arguments.first(where: {
        $0.label?.text == "additionalProperties"
      })
    else {
      return nil
    }
    return Self.parseSchemaSpecifier(from: additionalPropertiesArgument.expression)
  }

  private static func serializedSchemaArguments(
    from arguments: LabeledExprListSyntax,
    convertRegexPatternLiteralToString: Bool
  ) -> String? {
    guard !arguments.isEmpty else { return nil }

    let serialized = arguments.map { argument in
      let value: String
      if convertRegexPatternLiteralToString,
        argument.label?.text == "pattern",
        let regexPattern = Self.regexPatternLiteralBody(from: argument.expression)
      {
        if regexPattern.hashCount > 0 {
          value = Self.rawStringLiteral(regexPattern.body, hashCount: regexPattern.hashCount)
        } else {
          value = Self.quotedStringLiteral(regexPattern.body)
        }
      } else {
        value = argument.expression.trimmedDescription
      }

      if let label = argument.label?.text {
        return "\(label): \(value)"
      }
      return value
    }

    return serialized.joined(separator: ", ")
  }

  private struct RegexPatternLiteral {
    let body: String
    let hashCount: Int
  }

  private static func regexPatternLiteralBody(from expression: ExprSyntax) -> RegexPatternLiteral? {
    let raw = expression.trimmedDescription
    guard !raw.isEmpty else { return nil }

    var hashCount = 0
    var index = raw.startIndex
    while index < raw.endIndex, raw[index] == "#" {
      hashCount += 1
      index = raw.index(after: index)
    }

    guard index < raw.endIndex, raw[index] == "/" else { return nil }

    let prefixEnd = raw.index(after: index)
    let suffix = "/" + String(repeating: "#", count: hashCount)
    guard raw.hasSuffix(suffix) else { return nil }

    let suffixStart = raw.index(raw.endIndex, offsetBy: -suffix.count)
    guard prefixEnd <= suffixStart else { return nil }

    let body = String(raw[prefixEnd..<suffixStart])
    return RegexPatternLiteral(body: body, hashCount: hashCount)
  }

  private static func quotedStringLiteral(_ value: String) -> String {
    let escaped =
      value
      .replacingOccurrences(of: "\\", with: "\\\\")
      .replacingOccurrences(of: "\"", with: "\\\"")
      .replacingOccurrences(of: "\n", with: "\\n")
    return "\"\(escaped)\""
  }

  private static func rawStringLiteral(_ value: String, hashCount: Int) -> String {
    let hashes = String(repeating: "#", count: hashCount)
    let escaped =
      value
      .replacingOccurrences(of: "\"", with: "\\\(hashes)\"")
      .replacingOccurrences(of: "\n", with: "\\\(hashes)n")
    return "\(hashes)\"\(escaped)\"\(hashes)"
  }

  private static func stringLiteralValue(from expression: ExprSyntax) -> String? {
    let argument = expression.trimmedDescription
    guard argument.count >= 2, argument.first == "\"", argument.last == "\"" else {
      return nil
    }
    return String(argument.dropFirst().dropLast())
  }

  private static func semanticSchemaExpression(
    for schemaKind: SemanticSchemaKind,
    arguments: String?,
    isOptional: Bool
  ) -> String {
    let callSuffix = arguments.map { "(\($0))" } ?? "()"

    switch (schemaKind, isOptional) {
    case (.string, false):
      return ".string\(callSuffix)"
    case (.string, true):
      return ".union(string: .string\(callSuffix), null: true)"
    case (.number, false):
      return ".number\(callSuffix)"
    case (.number, true):
      return ".union(number: .number\(callSuffix), null: true)"
    case (.integer, false):
      return ".integer\(callSuffix)"
    case (.integer, true):
      return ".union(integer: .integer\(callSuffix), null: true)"
    case (.boolean, false):
      return ".bool()"
    case (.boolean, true):
      return ".union(bool: true, null: true)"
    }
  }

  private static func needleGenerationSchemaPropertyAttributes(
    in variableDecl: VariableDeclSyntax
  ) -> [AttributeSyntax] {
    variableDecl.attributes
      .compactMap { $0.as(AttributeSyntax.self) }
      .filter {
        let name = $0.attributeName.trimmedDescription
        return name == "NeedleGuide" || name == "Needle.NeedleGuide"
      }
  }

  private static func singleNeedleGuideAttribute(
    in variableDecl: VariableDeclSyntax
  ) -> AttributeSyntax? {
    let attributes = Self.needleGenerationSchemaPropertyAttributes(in: variableDecl)
    guard attributes.count == 1 else { return nil }
    return attributes.first
  }

  private static func isAttributeNamed(
    _ attribute: AttributeSyntax,
    names: Set<String>
  ) -> Bool {
    let attributeName = attribute.attributeName.trimmedDescription
    return names.contains(attributeName)
      || names.contains { attributeName.hasSuffix(".\($0)") }
  }

  private static func keyArgumentValue(in attribute: AttributeSyntax) -> String? {
    guard case .argumentList(let arguments) = attribute.arguments else { return nil }

    guard let keyExpression = arguments.first(where: { $0.label?.text == "key" })?.expression else {
      return nil
    }

    return Self.stringLiteralValue(from: keyExpression)
  }

  private static func isValid(
    type baseTypeName: String,
    for schemaKind: SemanticSchemaKind
  ) -> Bool {
    switch schemaKind {
    case .string:
      Self.stringSemanticTypes.contains(baseTypeName)
    case .number:
      Self.numberSemanticTypes.contains(baseTypeName)
    case .integer:
      Self.integerSemanticTypes.contains(baseTypeName)
    case .boolean:
      Self.booleanSemanticTypes.contains(baseTypeName)
    }
  }

  private static func diagnoseInvalidSemanticSchemaType(
    in attribute: AttributeSyntax,
    propertyName: String,
    typeName: String,
    expected: String,
    kind: String,
    context: some MacroExpansionContext
  ) {
    let message =
      "@NeedleGuide(\(kind)) can only be applied to properties of type \(expected). Found '\(propertyName): \(typeName)'."
    context.diagnose(Diagnostic(node: attribute, message: MacroExpansionErrorMessage(message)))
  }

  private static func diagnoseInvalidContainerSchemaType(
    in attribute: AttributeSyntax,
    propertyName: String,
    typeName: String,
    expected: String,
    kind: String,
    context: some MacroExpansionContext
  ) {
    let message =
      "@NeedleGuide(\(kind)) can only be applied to \(expected) properties. Found '\(propertyName): \(typeName)'."
    context.diagnose(Diagnostic(node: attribute, message: MacroExpansionErrorMessage(message)))
  }

  private static func diagnoseDictionarySchemaRequiresStringKeys(
    in attribute: AttributeSyntax,
    propertyName: String,
    typeName: String,
    context: some MacroExpansionContext
  ) {
    context.diagnose(
      Diagnostic(
        node: attribute,
        message: MacroExpansionErrorMessage(
          "@NeedleGuide(.object) can only be applied to dictionary properties with String keys. Found '\(propertyName): \(typeName)'."
        )
      )
    )
  }

  private static func diagnoseMissingTypeAnnotation(
    in binding: PatternBindingSyntax,
    context: some MacroExpansionContext
  ) {
    context.diagnose(
      Diagnostic(
        node: binding,
        message: MacroExpansionErrorMessage(
          "Stored properties must declare an explicit type."
        )
      )
    )
  }

  private static func diagnoseUnsupportedNeedleGenerationSchemaMember(
    in variableDecl: VariableDeclSyntax,
    context: some MacroExpansionContext
  ) {
    context.diagnose(
      Diagnostic(
        node: variableDecl,
        message: MacroExpansionErrorMessage(
          "Only stored properties are supported."
        )
      )
    )
  }

  private static func diagnoseConflictingSchemaAttributes(
    in attribute: AttributeSyntax,
    propertyName: String,
    context: some MacroExpansionContext
  ) {
    context.diagnose(
      Diagnostic(
        node: attribute,
        message: MacroExpansionErrorMessage(
          "@NeedleIgnored cannot be combined with @NeedleGuide on the same property."
        )
      )
    )
  }

  private static func diagnoseInvalidNeedleIgnoredAttribute(
    in attribute: AttributeSyntax,
    context: some MacroExpansionContext
  ) {
    context.diagnose(
      Diagnostic(
        node: attribute,
        message: MacroExpansionErrorMessage(
          "@NeedleIgnored can only be applied to optional properties or properties with default values."
        )
      )
    )
  }

  private static func isComputedProperty(_ binding: PatternBindingSyntax) -> Bool {
    guard let accessorBlock = binding.accessorBlock else { return false }
    return switch accessorBlock.accessors {
    case .getter:
      true
    case .accessors(let accessors):
      accessors.contains { accessor in
        switch accessor.accessorSpecifier.tokenKind {
        case .keyword(.get), .keyword(.set):
          true
        default:
          false
        }
      }
    }
  }

  private static func needleGenerationSchemaIgnoredAttribute(
    in variableDecl: VariableDeclSyntax
  ) -> AttributeSyntax? {
    variableDecl.attributes
      .compactMap { $0.as(AttributeSyntax.self) }
      .first {
        let name = $0.attributeName.trimmedDescription
        return name == "NeedleIgnored" || name == "Needle.NeedleIgnored"
      }
  }

  private static func needleGenerationSchemaPropertyAttribute(
    in variableDecl: VariableDeclSyntax,
    propertyName: String,
    context: some MacroExpansionContext
  ) -> AttributeSyntax? {
    let attributes = Self.needleGenerationSchemaPropertyAttributes(in: variableDecl)

    guard attributes.count <= 1 else {
      if let duplicateAttribute = attributes.dropFirst().first {
        Self.diagnoseDuplicateNeedleGuideAttribute(
          in: duplicateAttribute,
          propertyName: propertyName,
          context: context
        )
      }
      return nil
    }

    return attributes.first
  }

  private static func diagnoseDuplicateNeedleGuideAttribute(
    in attribute: AttributeSyntax,
    propertyName: String,
    context: some MacroExpansionContext
  ) {
    context.diagnose(
      Diagnostic(
        node: attribute,
        message: MacroExpansionErrorMessage(
          "Only one @NeedleGuide attribute can be applied to a stored property."
        )
      )
    )
  }

  private static func diagnoseInvalidNeedleGuideAttribute(
    in attribute: AttributeSyntax,
    propertyName: String,
    context: some MacroExpansionContext
  ) {
    context.diagnose(
      Diagnostic(
        node: attribute,
        message: MacroExpansionErrorMessage(
          "@NeedleGuide must be declared as @NeedleGuide(_NeedleGuideSchema.<kind>, key: \"schema_key\", description: \"text\") on '\(propertyName)'."
        )
      )
    )
  }
}
