// MARK: - TemplateParser

struct TemplateParser {
  private let source: [UInt8]
  private let options: EdgeToolsTemplate.Options
  private var position = 0

  private init(source: [UInt8], options: EdgeToolsTemplate.Options) {
    self.source = source
    self.options = options
  }

  static func parse(
    _ source: [UInt8],
    options: EdgeToolsTemplate.Options
  ) throws(TemplateFailure) -> TemplateNode {
    var parser = TemplateParser(source: source, options: options)
    let tokens = try parser.tokenize()
    var index = 0
    return try parser.parseTemplate(tokens, &index, isFully: true)
  }
}

// MARK: - Tokens

private enum SpaceHandling {
  case keep
  case strip

  init(_ marker: UInt8?) {
    self = marker == UInt8(ascii: "-") ? .strip : .keep
  }
}

private struct TemplateToken {
  enum Kind {
    case text([UInt8])
    case expression(TemplateExpression)
    case `if`(TemplateExpression)
    case elif(TemplateExpression)
    case `else`
    case endIf
    case `for`(
      names: [String],
      iterable: TemplateExpression,
      condition: TemplateExpression?,
      isRecursive: Bool
    )
    case endFor
    case generation
    case endGeneration
    case set(namespace: String?, names: [String], value: TemplateExpression?)
    case endSet
    case comment
    case macro(name: String, parameters: [TemplateMacro.Parameter])
    case endMacro
    case filter(TemplateExpression)
    case endFilter
    case loopControl(TemplateLoopControl)
    case call(TemplateExpression)
    case endCall
  }

  let position: Int
  let preSpace: SpaceHandling
  let postSpace: SpaceHandling
  let kind: Kind

  var typeName: String {
    switch self.kind {
    case .text: "text"
    case .expression: "expression"
    case .if: "if"
    case .elif: "elif"
    case .else: "else"
    case .endIf: "endif"
    case .for: "for"
    case .endFor: "endfor"
    case .generation: "generation"
    case .endGeneration: "endgeneration"
    case .set: "set"
    case .endSet: "endset"
    case .comment: "comment"
    case .macro: "macro"
    case .endMacro: "endmacro"
    case .filter: "filter"
    case .endFilter: "endfilter"
    case .loopControl: "break"
    case .call: "call"
    case .endCall: "endcall"
    }
  }

  var isExpression: Bool {
    if case .expression = self.kind {
      return true
    }
    return false
  }

  var unexpected: TemplateFailure {
    TemplateFailure("Unexpected " + self.typeName, position: self.position)
  }

  var unterminated: TemplateFailure {
    TemplateFailure("Unterminated " + self.typeName, position: self.position)
  }
}

// MARK: - Tokenizing

extension TemplateParser {
  private mutating func tokenize() throws(TemplateFailure) -> [TemplateToken] {
    var tokens = [TemplateToken]()
    do throws(TemplateFailure) {
      while self.position < self.source.count {
        tokens.append(try self.nextToken())
      }
    } catch {
      throw error.located(at: self.position)
    }
    return tokens
  }

  private mutating func nextToken() throws(TemplateFailure) -> TemplateToken {
    let location = self.position
    if let (preSpace, postSpace) = self.consumeComment() {
      return TemplateToken(
        position: location,
        preSpace: preSpace,
        postSpace: postSpace,
        kind: .comment
      )
    }
    if let preSpace = self.consumeTagOpening("{{", skippingSpaces: false) {
      let expression = try self.parseExpression()
      guard let postSpace = self.consumeTagClosing("}}") else {
        throw TemplateFailure("Expected closing expression tag")
      }
      return TemplateToken(
        position: location,
        preSpace: preSpace,
        postSpace: postSpace,
        kind: .expression(expression)
      )
    }
    if let preSpace = self.consumeTagOpening("{%", skippingSpaces: true) {
      let (kind, postSpace) = try self.parseBlock()
      return TemplateToken(position: location, preSpace: preSpace, postSpace: postSpace, kind: kind)
    }
    let textEnd = self.nextTagOpening() ?? self.source.count
    guard textEnd > self.position else {
      throw TemplateFailure(
        self.source[self.position + 1] == UInt8(ascii: "#")
          ? "Missing end of comment tag"
          : "Internal error: Expected a comment"
      )
    }
    let text = Array(self.source[self.position..<textEnd])
    self.position = textEnd
    return TemplateToken(position: location, preSpace: .keep, postSpace: .keep, kind: .text(text))
  }

  private mutating func parseBlock() throws(TemplateFailure) -> (TemplateToken.Kind, SpaceHandling)
  {
    guard let keyword = self.consumeBlockKeyword() else {
      throw TemplateFailure("Expected block keyword")
    }
    let kind: TemplateToken.Kind
    switch keyword {
    case "if":
      kind = .if(try self.parseExpression())
    case "elif":
      kind = .elif(try self.parseExpression())
    case "else":
      kind = .else
    case "endif":
      kind = .endIf
    case "for":
      let names = try self.parseVariableNames()
      guard self.consumeKeyword("in") else {
        throw TemplateFailure("Expected 'in' keyword in for block")
      }
      let iterable = try self.parseExpression(allowsConditional: false)
      let condition = try self.consumeKeyword("if") ? self.parseExpression() : nil
      kind = .for(
        names: names,
        iterable: iterable,
        condition: condition,
        isRecursive: self.consumeKeyword("recursive")
      )
    case "endfor":
      kind = .endFor
    case "generation":
      kind = .generation
    case "endgeneration":
      kind = .endGeneration
    case "set":
      if let (namespace, name) = self.consumeNamespacedName() {
        guard self.consume("=") else {
          throw TemplateFailure("Expected equals sign in set block")
        }
        kind = .set(namespace: namespace, names: [name], value: try self.parseExpression())
      } else {
        let names = try self.parseVariableNames()
        let value = try self.consume("=") ? self.parseExpression() : nil
        kind = .set(namespace: nil, names: names, value: value)
      }
    case "endset":
      kind = .endSet
    case "macro":
      guard let name = self.parseIdentifier()?.variableName else {
        throw TemplateFailure("Expected macro name in macro block")
      }
      kind = .macro(name: name, parameters: try self.parseParameters())
    case "endmacro":
      kind = .endMacro
    case "call":
      kind = .call(try self.parseExpression())
    case "endcall":
      kind = .endCall
    case "filter":
      kind = .filter(try self.parseExpression())
    case "endfilter":
      kind = .endFilter
    case "break":
      kind = .loopControl(.break)
    case "continue":
      kind = .loopControl(.continue)
    default:
      throw TemplateFailure("Unexpected block: " + keyword)
    }
    guard let postSpace = self.consumeTagClosing("%}") else {
      throw TemplateFailure("Expected closing block tag")
    }
    return (kind, postSpace)
  }

  private mutating func consumeComment() -> (SpaceHandling, SpaceHandling)? {
    guard self.peek("{#") else {
      return nil
    }
    var contentStart = self.position + 2
    let preSpace = SpaceHandling(self.spaceMarker(at: contentStart))
    if self.spaceMarker(at: contentStart) != nil {
      contentStart += 1
    }
    guard let close = self.firstIndex(of: "#}", from: contentStart) else {
      return nil
    }
    let postMarker = close > contentStart ? self.spaceMarker(at: close - 1) : nil
    self.position = close + 2
    return (preSpace, SpaceHandling(postMarker))
  }

  private mutating func consumeTagOpening(_ opening: String, skippingSpaces: Bool) -> SpaceHandling?
  {
    guard self.peek(opening) else {
      return nil
    }
    self.position += 2
    let marker = self.spaceMarker(at: self.position)
    if marker != nil {
      self.position += 1
    }
    if skippingSpaces {
      self.skipSpaces()
    }
    return SpaceHandling(marker)
  }

  private mutating func consumeTagClosing(_ closing: String) -> SpaceHandling? {
    let start = self.position
    self.skipSpaces()
    let marker = self.spaceMarker(at: self.position)
    let closingStart = self.position + (marker == nil ? 0 : 1)
    guard self.peek(closing, at: closingStart) else {
      self.position = start
      return nil
    }
    self.position = closingStart + 2
    return SpaceHandling(marker)
  }

  private mutating func consumeBlockKeyword() -> String? {
    let start = self.position
    self.skipSpaces()
    let word = self.word(at: self.position)
    guard blockKeywords.contains(word) else {
      self.position = start
      return nil
    }
    self.position += word.utf8.count
    return word
  }

  private func nextTagOpening() -> Int? {
    var index = self.position
    while index + 1 < self.source.count {
      let next = self.source[index + 1]
      if self.source[index] == UInt8(ascii: "{")
        && (next == UInt8(ascii: "{") || next == UInt8(ascii: "%") || next == UInt8(ascii: "#"))
      {
        return index
      }
      index += 1
    }
    return nil
  }

  private func spaceMarker(at index: Int) -> UInt8? {
    guard index < self.source.count,
      self.source[index] == UInt8(ascii: "-") || self.source[index] == UInt8(ascii: "~")
    else {
      return nil
    }
    return self.source[index]
  }
}

// MARK: - Template Structure

extension TemplateParser {
  private func parseTemplate(
    _ tokens: [TemplateToken],
    _ index: inout Int,
    isFully: Bool = false
  ) throws(TemplateFailure) -> TemplateNode {
    var children = [TemplateNode]()
    parsing: while index < tokens.count {
      let start = index
      let token = tokens[index]
      index += 1
      func parseBody(closedBy isClosing: (TemplateToken.Kind) -> Bool) throws(TemplateFailure)
        -> TemplateNode
      {
        let body = try self.parseTemplate(tokens, &index)
        guard index < tokens.count, isClosing(tokens[index].kind) else {
          throw tokens[start].unterminated
        }
        index += 1
        return body
      }
      switch token.kind {
      case .if(let condition):
        var branches = [(condition: TemplateExpression?, body: TemplateNode)]()
        branches.append((condition, try self.parseTemplate(tokens, &index)))
        while index < tokens.count, case .elif(let condition) = tokens[index].kind {
          index += 1
          branches.append((condition, try self.parseTemplate(tokens, &index)))
        }
        if index < tokens.count, case .else = tokens[index].kind {
          index += 1
          branches.append((nil, try self.parseTemplate(tokens, &index)))
        }
        guard index < tokens.count, case .endIf = tokens[index].kind else {
          throw tokens[start].unterminated
        }
        index += 1
        children.append(TemplateNode(.conditional(branches), at: token.position))
      case .for(let names, let iterable, let condition, let isRecursive):
        let body = try self.parseTemplate(tokens, &index)
        var elseBody: TemplateNode?
        if index < tokens.count, case .else = tokens[index].kind {
          index += 1
          elseBody = try self.parseTemplate(tokens, &index)
        }
        guard index < tokens.count, case .endFor = tokens[index].kind else {
          throw tokens[start].unterminated
        }
        index += 1
        let loop = TemplateLoop(
          names: names,
          iterable: iterable,
          condition: condition,
          body: body,
          isRecursive: isRecursive,
          elseBody: elseBody
        )
        children.append(TemplateNode(.loop(loop), at: token.position))
      case .generation:
        children.append(
          try parseBody {
            if case .endGeneration = $0 { true } else { false }
          }
        )
      case .text(let text):
        children.append(self.textNode(text, tokens, start: start, end: index, at: token.position))
      case .expression(let expression):
        children.append(TemplateNode(.expression(expression), at: token.position))
      case .set(let namespace, let names, let value?):
        children.append(
          TemplateNode(.set(namespace: namespace, names: names, value: value), at: token.position)
        )
      case .set(let namespace, let names, nil):
        let body = try parseBody {
          if case .endSet = $0 { true } else { false }
        }
        guard namespace == nil else {
          throw TemplateFailure("Namespaced set not supported in set with template value")
        }
        guard names.count == 1 else {
          throw TemplateFailure("Structural assignment not supported in set with template value")
        }
        children.append(TemplateNode(.setBlock(name: names[0], body: body), at: token.position))
      case .macro(let name, let parameters):
        let body = try parseBody {
          if case .endMacro = $0 { true } else { false }
        }
        let macro = TemplateMacro(name: name, parameters: parameters, body: body)
        children.append(TemplateNode(.macro(macro), at: token.position))
      case .call(let expression):
        let body = try parseBody {
          if case .endCall = $0 { true } else { false }
        }
        children.append(TemplateNode(.call(expression, body: body), at: token.position))
      case .filter(let filter):
        let body = try parseBody {
          if case .endFilter = $0 { true } else { false }
        }
        children.append(TemplateNode(.filter(filter, body: body), at: token.position))
      case .comment:
        break
      case .loopControl(let control):
        children.append(TemplateNode(.loopControl(control), at: token.position))
      case .endFor, .endSet, .endMacro, .endCall, .endFilter, .endIf, .else, .endGeneration,
        .elif:
        index -= 1
        break parsing
      }
    }
    if isFully && index < tokens.count {
      throw tokens[index].unexpected
    }
    switch children.count {
    case 0: return TemplateNode(.text(""), at: 0)
    case 1: return children[0]
    default: return TemplateNode(.sequence(children), at: children[0].position)
    }
  }

  private func textNode(
    _ text: [UInt8],
    _ tokens: [TemplateToken],
    start: Int,
    end: Int,
    at position: Int
  ) -> TemplateNode {
    var text = text
    let preSpace = start > 0 ? tokens[start - 1].postSpace : .keep
    let postSpace = end < tokens.count ? tokens[end].preSpace : .keep
    if postSpace == .strip {
      text.removeLast(text.reversed().prefix { $0.isTemplateSpace }.count)
    } else if self.options.lstripBlocks && end < tokens.count {
      let lineEnd = text.count - text.reversed().prefix { $0 == 0x20 || $0 == 0x09 }.count
      if (lineEnd == 0 && start == 0) || (lineEnd > 0 && text[lineEnd - 1] == .newline) {
        text.removeSubrange(lineEnd...)
      }
    }
    if preSpace == .strip {
      text.removeFirst(text.prefix { $0.isTemplateSpace }.count)
    } else if self.options.trimBlocks && start > 0 && !tokens[start - 1].isExpression
      && text.first == .newline
    {
      text.removeFirst()
    }
    if end == tokens.count && !self.options.keepTrailingNewline && text.last == .newline {
      text.removeLast()
      if text.last == UInt8(ascii: "\r") {
        text.removeLast()
      }
    }
    return TemplateNode(.text(String(decoding: text, as: UTF8.self)), at: position)
  }
}

// MARK: - Expressions

extension TemplateParser {
  private mutating func parseExpression(
    allowsConditional: Bool = true
  ) throws(TemplateFailure) -> TemplateExpression {
    let left = try self.parseLogicalOr()
    guard self.position < self.source.count, allowsConditional, self.consumeKeyword("if") else {
      return left
    }
    let location = self.position
    let condition = try self.parseLogicalOr()
    let otherwise = try self.consumeKeyword("else") ? self.parseExpression() : nil
    return TemplateExpression(.conditional(condition, then: left, else: otherwise), at: location)
  }

  private mutating func parseLogicalOr() throws(TemplateFailure) -> TemplateExpression {
    var left = try self.parseLogicalAnd()
    let location = self.position
    while self.consumeKeyword("or") {
      left = TemplateExpression(.binary(.or, left, try self.parseLogicalAnd()), at: location)
    }
    return left
  }

  private mutating func parseLogicalAnd() throws(TemplateFailure) -> TemplateExpression {
    var left = try self.parseLogicalNot()
    let location = self.position
    while self.consumeKeyword("and") {
      left = TemplateExpression(.binary(.and, left, try self.parseLogicalNot()), at: location)
    }
    return left
  }

  private mutating func parseLogicalNot() throws(TemplateFailure) -> TemplateExpression {
    let location = self.position
    guard self.consumeKeyword("not") else {
      return try self.parseLogicalCompare()
    }
    return TemplateExpression(.unary(.not, try self.parseLogicalNot()), at: location)
  }

  private mutating func parseLogicalCompare() throws(TemplateFailure) -> TemplateExpression {
    var left = try self.parseStringConcat()
    while let comparison = self.consumeComparisonOperator() {
      guard comparison != .is else {
        let isNegated = self.consumeKeyword("not")
        guard let identifier = self.parseIdentifier() else {
          throw TemplateFailure("Expected identifier after 'is' keyword")
        }
        return TemplateExpression(
          .binary(isNegated ? .isNot : .is, left, identifier),
          at: left.position
        )
      }
      let right = try self.parseStringConcat()
      left = TemplateExpression(.binary(comparison, left, right), at: self.position)
    }
    return left
  }

  private mutating func parseStringConcat() throws(TemplateFailure) -> TemplateExpression {
    let left = try self.parseMathPow()
    guard self.consumeOperator("~", unlessFollowedBy: ["}"]) else {
      return left
    }
    let right = try self.parseLogicalAnd()
    return TemplateExpression(.binary(.concatenate, left, right), at: self.position)
  }

  private mutating func parseMathPow() throws(TemplateFailure) -> TemplateExpression {
    var left = try self.parseMathPlusMinus()
    while self.consume("**") {
      let right = try self.parseMathPlusMinus()
      left = TemplateExpression(.binary(.power, left, right), at: self.position)
    }
    return left
  }

  private mutating func parseMathPlusMinus() throws(TemplateFailure) -> TemplateExpression {
    var left = try self.parseMathMulDiv()
    while let isPlus = self.consumePlusOrMinus() {
      let right = try self.parseMathMulDiv()
      left = TemplateExpression(.binary(isPlus ? .add : .subtract, left, right), at: self.position)
    }
    return left
  }

  private mutating func parseMathMulDiv() throws(TemplateFailure) -> TemplateExpression {
    var left = try self.parseMathUnaryPlusMinus()
    while let multiplicative = self.consumeMultiplicativeOperator() {
      let right = try self.parseMathUnaryPlusMinus()
      left = TemplateExpression(.binary(multiplicative, left, right), at: self.position)
    }
    guard self.consume("|") else {
      return left
    }
    let filter = try self.parseMathMulDiv()
    guard case .filter(let parts) = filter.kind else {
      return TemplateExpression(.filter([left, filter]), at: self.position)
    }
    return TemplateExpression(.filter([left] + parts), at: filter.position)
  }

  private mutating func parseMathUnaryPlusMinus() throws(TemplateFailure) -> TemplateExpression {
    let sign = self.consumePlusOrMinus()
    let expression = try self.parseExpansion()
    guard let isPlus = sign else {
      return expression
    }
    return TemplateExpression(.unary(isPlus ? .plus : .minus, expression), at: self.position)
  }

  private mutating func parseExpansion() throws(TemplateFailure) -> TemplateExpression {
    let isDictionary = self.consume("**")
    let isExpansion = isDictionary || self.consume("*")
    let expression = try self.parseValueExpression()
    guard isExpansion else {
      return expression
    }
    return TemplateExpression(
      .unary(isDictionary ? .dictionaryExpansion : .expansion, expression),
      at: self.position
    )
  }

  private mutating func parseValueExpression() throws(TemplateFailure) -> TemplateExpression {
    var value = try self.parseValue()
    while self.position < self.source.count {
      self.skipSpaces()
      if self.consume("[") {
        let index = try self.parseSubscriptIndex()
        value = TemplateExpression(.subscripted(value, index: index), at: value.position)
      } else if self.consume(".") {
        guard let identifier = self.parseIdentifier(), let name = identifier.variableName else {
          throw TemplateFailure("Expected identifier in subscript")
        }
        self.skipSpaces()
        if self.peek("(") {
          let arguments = try self.parseCallArguments()
          value = TemplateExpression(
            .methodCall(value, method: name, arguments),
            at: identifier.position
          )
        } else {
          let key = TemplateExpression(.literal(.string(name)), at: identifier.position)
          value = TemplateExpression(.subscripted(value, index: key), at: identifier.position)
        }
      } else {
        break
      }
      self.skipSpaces()
    }
    guard self.peek("(") else {
      return value
    }
    let location = self.position
    return TemplateExpression(.call(value, try self.parseCallArguments()), at: location)
  }

  private mutating func parseSubscriptIndex() throws(TemplateFailure) -> TemplateExpression {
    let location = self.position
    let start = try self.peek(":") ? nil : self.parseExpression()
    var index = start
    if self.consume(":") {
      let end = try self.peek(":") || self.peek("]") ? nil : self.parseExpression()
      var step: TemplateExpression?
      if self.consume(":") && !self.peek("]") {
        step = try self.parseExpression()
      }
      index = TemplateExpression(.slice(start: start, end: end, step: step), at: location)
    }
    guard let index else {
      throw TemplateFailure("Empty index in subscript")
    }
    guard self.consume("]") else {
      throw TemplateFailure("Expected closing bracket in subscript")
    }
    return index
  }

  private mutating func parseValue() throws(TemplateFailure) -> TemplateExpression {
    let location = self.position
    if let constant = try self.parseConstant() {
      return TemplateExpression(.literal(constant), at: location)
    }
    if self.consumeKeyword("null") {
      return TemplateExpression(.literal(.null), at: location)
    }
    if let identifier = self.parseIdentifier() {
      return identifier
    }
    if let braced = try self.parseBracedExpressionOrArray() {
      return braced
    }
    if let array = try self.parseArray() {
      return array
    }
    if let dictionary = try self.parseDictionary() {
      return dictionary
    }
    throw TemplateFailure("Expected value expression")
  }

  private mutating func parseBracedExpressionOrArray() throws(TemplateFailure)
    -> TemplateExpression?
  {
    guard self.consume("(") else {
      return nil
    }
    let expression = try self.parseExpression()
    if self.consume(")") {
      return expression
    }
    var tuple = [expression]
    while self.position < self.source.count {
      guard self.consume(",") else {
        throw TemplateFailure("Expected comma in tuple")
      }
      tuple.append(try self.parseExpression())
      if self.consume(")") {
        return TemplateExpression(.array(tuple), at: self.position)
      }
    }
    throw TemplateFailure("Expected closing parenthesis")
  }

  private mutating func parseArray() throws(TemplateFailure) -> TemplateExpression? {
    guard self.consume("[") else {
      return nil
    }
    guard !self.consume("]") else {
      return TemplateExpression(.array([]), at: self.position)
    }
    var elements = [try self.parseExpression()]
    while self.position < self.source.count {
      if self.consume(",") {
        elements.append(try self.parseExpression())
      } else if self.consume("]") {
        return TemplateExpression(.array(elements), at: self.position)
      } else {
        throw TemplateFailure("Expected comma or closing bracket in array")
      }
    }
    throw TemplateFailure("Expected closing bracket")
  }

  private mutating func parseDictionary() throws(TemplateFailure) -> TemplateExpression? {
    guard self.consume("{") else {
      return nil
    }
    guard !self.consume("}") else {
      return TemplateExpression(.dictionary([]), at: self.position)
    }
    var entries = [try self.parseDictionaryEntry()]
    while self.position < self.source.count {
      if self.consume(",") {
        entries.append(try self.parseDictionaryEntry())
      } else if self.consume("}") {
        return TemplateExpression(.dictionary(entries), at: self.position)
      } else {
        throw TemplateFailure("Expected comma or closing brace in dictionary")
      }
    }
    throw TemplateFailure("Expected closing brace")
  }

  private mutating func parseDictionaryEntry() throws(TemplateFailure)
    -> (key: TemplateExpression, value: TemplateExpression)
  {
    let key = try self.parseExpression()
    guard self.consume(":") else {
      throw TemplateFailure("Expected colon betweek key & value in dictionary")
    }
    return (key, try self.parseExpression())
  }

  private mutating func parseCallArguments() throws(TemplateFailure) -> TemplateCallArguments {
    var arguments = TemplateCallArguments()
    try self.parseArgumentList(opening: "Expected opening parenthesis in call args") {
      expression,
      parser throws(TemplateFailure) in
      if let name = expression.variableName, parser.consume("=") {
        arguments.named.append((name, try parser.parseExpression()))
      } else {
        arguments.positional.append(expression)
      }
    }
    return arguments
  }

  private mutating func parseParameters() throws(TemplateFailure) -> [TemplateMacro.Parameter] {
    var parameters = [TemplateMacro.Parameter]()
    try self.parseArgumentList(opening: "Expected opening parenthesis in param list") {
      expression,
      parser throws(TemplateFailure) in
      guard let name = expression.variableName else {
        parameters.append(TemplateMacro.Parameter(name: "", defaultValue: expression))
        return
      }
      let defaultValue = try parser.consume("=") ? parser.parseExpression() : nil
      parameters.append(TemplateMacro.Parameter(name: name, defaultValue: defaultValue))
    }
    return parameters
  }

  private mutating func parseArgumentList(
    opening openingMessage: String,
    _ parseArgument: (TemplateExpression, inout TemplateParser) throws(TemplateFailure) -> Void
  ) throws(TemplateFailure) {
    self.skipSpaces()
    guard self.consume("(") else {
      throw TemplateFailure(openingMessage)
    }
    while self.position < self.source.count {
      if self.consume(")") {
        return
      }
      try parseArgument(try self.parseExpression(), &self)
      guard self.consume(",") else {
        guard self.consume(")") else {
          throw TemplateFailure("Expected closing parenthesis in call args")
        }
        return
      }
    }
    throw TemplateFailure("Expected closing parenthesis in call args")
  }

  private mutating func parseIdentifier() -> TemplateExpression? {
    let location = self.position
    let start = self.position
    self.skipSpaces()
    let word = self.word(at: self.position)
    guard let first = word.utf8.first, first.isASCIILetter || first == UInt8(ascii: "_"),
      !reservedWords.contains(word)
    else {
      self.position = start
      return nil
    }
    self.position += word.utf8.count
    return TemplateExpression(.variable(word), at: location)
  }

  private mutating func parseVariableNames() throws(TemplateFailure) -> [String] {
    let start = self.position
    self.skipSpaces()
    guard let first = self.consumeWord() else {
      self.position = start
      throw TemplateFailure("Expected variable names")
    }
    var names = [first]
    while true {
      let mark = self.position
      self.skipSpaces()
      guard self.peek(",") else {
        self.position = mark
        break
      }
      self.position += 1
      self.skipSpaces()
      guard let name = self.consumeWord() else {
        self.position = mark
        break
      }
      names.append(name)
    }
    self.skipSpaces()
    return names
  }

  private mutating func consumeNamespacedName() -> (String, String)? {
    let start = self.position
    self.skipSpaces()
    guard let namespace = self.consumeWord() else {
      self.position = start
      return nil
    }
    self.skipSpaces()
    guard self.peek(".") else {
      self.position = start
      return nil
    }
    self.position += 1
    self.skipSpaces()
    guard let name = self.consumeWord() else {
      self.position = start
      return nil
    }
    return (namespace, name)
  }
}

// MARK: - Constants

extension TemplateParser {
  private mutating func parseConstant() throws(TemplateFailure) -> TemplatePrimitive? {
    let start = self.position
    self.skipSpaces()
    guard self.position < self.source.count else {
      return nil
    }
    if self.peek("\"") || self.peek("'"), var string = self.parseString() {
      // jinja2 concatenates adjacent string literals the way Python does (`"a" "b"` is `"ab"`),
      // which chat templates use to wrap long messages.
      while true {
        let mark = self.position
        self.skipSpaces()
        guard self.peek("\"") || self.peek("'"), let adjacent = self.parseString() else {
          self.position = mark
          break
        }
        string += adjacent
      }
      return .string(string)
    }
    if self.consumeKeyword("true") || self.consumeKeyword("True") {
      return .boolean(true)
    }
    if self.consumeKeyword("false") || self.consumeKeyword("False") {
      return .boolean(false)
    }
    if self.consumeKeyword("None") {
      return .null
    }
    if let number = try self.parseNumber() {
      return number
    }
    self.position = start
    return nil
  }

  private mutating func parseString() -> String? {
    self.skipSpaces()
    guard self.peek("\"") || self.peek("'") else {
      return nil
    }
    let quote = self.source[self.position]
    var bytes = [UInt8]()
    var isEscaped = false
    self.position += 1
    while self.position < self.source.count {
      let byte = self.source[self.position]
      self.position += 1
      if isEscaped {
        isEscaped = false
        bytes.append(escapedByte(byte))
      } else if byte == UInt8(ascii: "\\") {
        isEscaped = true
      } else if byte == quote {
        return String(decoding: bytes, as: UTF8.self)
      } else {
        bytes.append(byte)
      }
    }
    return nil
  }

  private mutating func parseNumber() throws(TemplateFailure) -> TemplatePrimitive? {
    let before = self.position
    self.skipSpaces()
    let start = self.position
    var hasDecimal = false
    var hasExponent = false
    if self.peek("-") || self.peek("+") {
      self.position += 1
    }
    while self.position < self.source.count {
      let byte = self.source[self.position]
      if byte.isASCIIDigit {
        self.position += 1
      } else if byte == UInt8(ascii: ".") {
        guard !hasDecimal else {
          throw TemplateFailure("Multiple decimal points")
        }
        hasDecimal = true
        self.position += 1
      } else if self.position != start && (byte == UInt8(ascii: "e") || byte == UInt8(ascii: "E")) {
        guard !hasExponent else {
          throw TemplateFailure("Multiple exponents")
        }
        hasExponent = true
        self.position += 1
      } else {
        break
      }
    }
    guard self.position != start else {
      self.position = before
      return nil
    }
    let text = self.source[start..<self.position]
    guard let number = jsonNumber(text) else {
      throw TemplateFailure("Failed to parse number: '\(String(decoding: text, as: UTF8.self))'")
    }
    return number
  }
}

// MARK: - Operators

extension TemplateParser {
  private mutating func consumeComparisonOperator() -> TemplateBinaryOperator? {
    let start = self.position
    self.skipSpaces()
    let symbols: [(String, TemplateBinaryOperator)] = [
      ("==", .equal), ("!=", .notEqual), ("<=", .lessOrEqual), ("<", .less),
      (">=", .greaterOrEqual), (">", .greater)
    ]
    if let (symbol, comparison) = symbols.first(where: { self.peek($0.0) }) {
      self.position += symbol.utf8.count
      return comparison
    }
    if self.consumeKeyword("in") {
      return .in
    }
    if self.consumeKeyword("is") {
      return .is
    }
    if self.peekKeyword("not") {
      self.position += 3
      let spacesStart = self.position
      self.skipSpaces()
      if self.position > spacesStart && self.peekKeyword("in") {
        self.position += 2
        return .notIn
      }
    }
    self.position = start
    return nil
  }

  private mutating func consumePlusOrMinus() -> Bool? {
    if self.consume("+") {
      return true
    }
    if self.consumeOperator("-", unlessFollowedBy: ["}}", "%}", "#}"]) {
      return false
    }
    return nil
  }

  private mutating func consumeMultiplicativeOperator() -> TemplateBinaryOperator? {
    if self.consume("**") {
      return .power
    }
    if self.consume("*") {
      return .multiply
    }
    if self.consume("//") {
      return .floorDivide
    }
    if self.consume("/") {
      return .divide
    }
    if self.consumeOperator("%", unlessFollowedBy: ["}"]) {
      return .remainder
    }
    return nil
  }

  private mutating func consumeOperator(_ symbol: String, unlessFollowedBy excluded: [String])
    -> Bool
  {
    let start = self.position
    self.skipSpaces()
    let next = self.position + symbol.utf8.count
    guard self.peek(symbol), !excluded.contains(where: { self.peek($0, at: next) }) else {
      self.position = start
      return false
    }
    self.position = next
    return true
  }
}

// MARK: - Scanning

extension TemplateParser {
  private mutating func skipSpaces() {
    while self.position < self.source.count && self.source[self.position].isTemplateSpace {
      self.position += 1
    }
  }

  private func peek(_ symbol: String) -> Bool {
    self.peek(symbol, at: self.position)
  }

  private func peek(_ symbol: String, at index: Int) -> Bool {
    let count = symbol.utf8.count
    guard index >= 0, index + count <= self.source.count else {
      return false
    }
    return self.source[index..<(index + count)].elementsEqual(symbol.utf8)
  }

  private func peekKeyword(_ keyword: String) -> Bool {
    let end = self.position + keyword.utf8.count
    return self.peek(keyword) && (end == self.source.count || !self.source[end].isTemplateWord)
  }

  private mutating func consume(_ symbol: String) -> Bool {
    let start = self.position
    self.skipSpaces()
    guard self.peek(symbol) else {
      self.position = start
      return false
    }
    self.position += symbol.utf8.count
    return true
  }

  private mutating func consumeKeyword(_ keyword: String) -> Bool {
    let start = self.position
    self.skipSpaces()
    guard self.peekKeyword(keyword) else {
      self.position = start
      return false
    }
    self.position += keyword.utf8.count
    return true
  }

  private mutating func consumeWord() -> String? {
    let word = self.word(at: self.position)
    guard !word.isEmpty else {
      return nil
    }
    self.position += word.utf8.count
    return word
  }

  private func word(at index: Int) -> String {
    let end = self.source[index...].firstIndex { !$0.isTemplateWord } ?? self.source.count
    return String(decoding: self.source[index..<end], as: UTF8.self)
  }

  private func firstIndex(of symbol: String, from start: Int) -> Int? {
    (start..<max(start, self.source.count)).first { self.peek(symbol, at: $0) }
  }
}

// MARK: - Helpers

private let blockKeywords: Set<String> = [
  "if", "else", "elif", "endif", "for", "endfor", "generation", "endgeneration", "set", "endset",
  "block", "endblock", "macro", "endmacro", "filter", "endfilter", "break", "continue", "call",
  "endcall"
]

private let reservedWords: Set<String> = ["not", "is", "and", "or", "del"]

private func escapedByte(_ byte: UInt8) -> UInt8 {
  switch byte {
  case UInt8(ascii: "n"): .newline
  case UInt8(ascii: "r"): UInt8(ascii: "\r")
  case UInt8(ascii: "t"): UInt8(ascii: "\t")
  case UInt8(ascii: "b"): 0x08
  case UInt8(ascii: "f"): 0x0C
  default: byte
  }
}

private func jsonNumber(_ text: ArraySlice<UInt8>) -> TemplatePrimitive? {
  var index = text.startIndex
  func consumeDigits() -> Int {
    let start = index
    while index < text.endIndex && text[index].isASCIIDigit {
      index += 1
    }
    return index - start
  }
  func consume(_ predicate: (UInt8) -> Bool) -> Bool {
    guard index < text.endIndex, predicate(text[index]) else {
      return false
    }
    index += 1
    return true
  }
  _ = consume { $0 == UInt8(ascii: "-") }
  guard consume({ $0 == UInt8(ascii: "0") }) || consumeDigits() > 0 else {
    return nil
  }
  var isInteger = true
  if consume({ $0 == UInt8(ascii: ".") }) {
    isInteger = false
    guard consumeDigits() > 0 else {
      return nil
    }
  }
  if consume({ $0 == UInt8(ascii: "e") || $0 == UInt8(ascii: "E") }) {
    isInteger = false
    _ = consume { $0 == UInt8(ascii: "+") || $0 == UInt8(ascii: "-") }
    guard consumeDigits() > 0 else {
      return nil
    }
  }
  guard index == text.endIndex else {
    return nil
  }
  let string = String(decoding: text, as: UTF8.self)
  if isInteger, let integer = Int64(string) {
    return .integer(integer)
  }
  return Double(string).map { .number($0) }
}
