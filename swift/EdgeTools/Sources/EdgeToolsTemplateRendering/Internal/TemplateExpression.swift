#if canImport(Darwin)
  import Darwin
#elseif canImport(Glibc)
  import Glibc
#elseif canImport(Musl)
  import Musl
#elseif canImport(WASILibc)
  import WASILibc
#elseif canImport(Android)
  import Android
#elseif canImport(ucrt)
  import ucrt
#endif

// MARK: - TemplateExpression

struct TemplateExpression: Sendable {
  let position: Int
  let kind: Kind

  indirect enum Kind: Sendable {
    case variable(String)
    case literal(TemplatePrimitive)
    case array([TemplateExpression])
    case dictionary([(key: TemplateExpression, value: TemplateExpression)])
    case slice(start: TemplateExpression?, end: TemplateExpression?, step: TemplateExpression?)
    case subscripted(TemplateExpression, index: TemplateExpression)
    case unary(TemplateUnaryOperator, TemplateExpression)
    case binary(TemplateBinaryOperator, TemplateExpression, TemplateExpression)
    case conditional(
      TemplateExpression,
      then: TemplateExpression,
      else: TemplateExpression?
    )
    case methodCall(TemplateExpression, method: String, TemplateCallArguments)
    case call(TemplateExpression, TemplateCallArguments)
    case filter([TemplateExpression])
  }

  init(_ kind: Kind, at position: Int) {
    self.position = position
    self.kind = kind
  }
}

enum TemplateUnaryOperator: Sendable {
  case plus
  case minus
  case not
  case expansion
  case dictionaryExpansion
}

enum TemplateBinaryOperator: Sendable {
  case concatenate
  case add
  case subtract
  case multiply
  case power
  case divide
  case floorDivide
  case remainder
  case equal
  case notEqual
  case less
  case greater
  case lessOrEqual
  case greaterOrEqual
  case and
  case or
  case `in`
  case notIn
  case `is`
  case isNot
}

struct TemplateCallArguments: Sendable {
  var positional = [TemplateExpression]()
  var named = [(name: String, value: TemplateExpression)]()
}

// MARK: - Evaluation

extension TemplateExpression {
  var variableName: String? {
    guard case .variable(let name) = self.kind else {
      return nil
    }
    return name
  }

  func evaluate(_ context: TemplateContext) throws(TemplateFailure) -> TemplateValue {
    try withTemplateLocation(self.position) { () throws(TemplateFailure) in
      try self.evaluateKind(context)
    }
  }

  private func evaluateKind(_ context: TemplateContext) throws(TemplateFailure) -> TemplateValue {
    switch self.kind {
    case .variable(let name):
      return context.value(named: name) ?? .null
    case .literal(let primitive):
      return primitive.value
    case .array(let elements):
      var values = [TemplateValue]()
      for element in elements {
        values.append(try element.evaluate(context))
      }
      return .array(values)
    case .dictionary(let entries):
      let object = TemplateValue.object(TemplateObject())
      for entry in entries {
        let key = try entry.key.evaluate(context)
        try object.setElement(try entry.value.evaluate(context), at: key)
      }
      return object
    case .slice:
      throw TemplateFailure("SliceExpr not implemented")
    case .subscripted(let base, let index):
      return try evaluateSubscript(base, index, context)
    case .unary(let unaryOperator, let operand):
      let value = try operand.evaluate(context)
      switch unaryOperator {
      case .plus: return value
      case .minus: return try value.negated()
      case .not: return .boolean(!value.isTruthy)
      case .expansion, .dictionaryExpansion:
        throw TemplateFailure(
          "Expansion operator is only supported in function calls and collections"
        )
      }
    case .binary(let binaryOperator, let lhs, let rhs):
      let left = try lhs.evaluate(context)
      guard left.isCallable else {
        return try binaryOperator.apply(left, rhs, context)
      }
      return .function { callContext, arguments throws(TemplateFailure) in
        try binaryOperator.apply(try left.call(callContext, arguments), rhs, context)
      }
    case .conditional(let condition, let then, let otherwise):
      if try condition.evaluate(context).isTruthy {
        return try then.evaluate(context)
      }
      return try otherwise?.evaluate(context) ?? .null
    case .methodCall(let object, let method, let arguments):
      let value = try object.evaluate(context)
      let arguments = try arguments.evaluate(context)
      guard !value.isNull else {
        throw TemplateFailure("Trying to call method '\(method)' on null")
      }
      return try callMethod(method, on: value, arguments, context)
    case .call(let callee, let arguments):
      let function = try callee.evaluate(context)
      guard function.isCallable else {
        throw TemplateFailure("Object is not callable: " + function.repr(indent: 2))
      }
      return try function.call(context, try arguments.evaluate(context))
    case .filter(let parts):
      return try evaluateFilter(parts, context)
    }
  }
}

extension TemplateCallArguments {
  func evaluate(_ context: TemplateContext) throws(TemplateFailure) -> TemplateArguments {
    var arguments = TemplateArguments()
    for argument in self.positional {
      switch argument.kind {
      case .unary(.expansion, let operand):
        guard case .array(let array) = try operand.evaluate(context) else {
          throw TemplateFailure("Expansion operator only supported on arrays")
        }
        arguments.positional += array.elements
      case .unary(.dictionaryExpansion, let operand):
        guard case .object(let object) = try operand.evaluate(context) else {
          throw TemplateFailure("ExpansionDict operator only supported on objects")
        }
        for (key, value) in object.entries {
          arguments.named.append((try key.value.stringValue(), value))
        }
      default:
        arguments.positional.append(try argument.evaluate(context))
      }
    }
    for argument in self.named {
      arguments.named.append((argument.name, try argument.value.evaluate(context)))
    }
    return arguments
  }
}

// MARK: - Operators

extension TemplateBinaryOperator {
  fileprivate func apply(
    _ left: TemplateValue,
    _ rhs: TemplateExpression,
    _ context: TemplateContext
  ) throws(TemplateFailure) -> TemplateValue {
    switch self {
    case .is, .isNot:
      guard let name = rhs.variableName else {
        throw TemplateFailure("Right side of 'is' operator must be a variable")
      }
      let matches = try templateTest(name, left)
      return .boolean(self == .is ? matches : !matches)
    case .and:
      guard left.isTruthy else {
        return .boolean(false)
      }
      return .boolean(try rhs.evaluate(context).isTruthy)
    case .or:
      guard !left.isTruthy else {
        return left
      }
      return try rhs.evaluate(context)
    default:
      break
    }
    let right = try rhs.evaluate(context)
    switch self {
    case .concatenate: return .string(left.displayString + right.displayString)
    case .add: return try left.adding(right)
    case .subtract: return try left.subtracting(right)
    case .multiply: return try left.multiplying(right)
    case .divide: return try left.dividing(right)
    case .power: return .number(pow(try left.doubleValue(), try right.doubleValue()))
    case .floorDivide:
      return .integer(try templateIntegerDivision(left.integerValue(), right.integerValue()))
    case .remainder:
      return .integer(try templateRemainder(left.integerValue(), right.integerValue()))
    case .equal: return .boolean(left.isEqual(to: right))
    case .notEqual: return .boolean(!left.isEqual(to: right))
    case .less: return .boolean(try left.isLess(than: right))
    case .greater: return .boolean(try left.isGreater(than: right))
    case .lessOrEqual: return .boolean(!(try left.isGreater(than: right)))
    case .greaterOrEqual: return .boolean(!(try left.isLess(than: right)))
    case .in: return .boolean(try templateIn(left, right))
    case .notIn: return .boolean(!(try templateIn(left, right)))
    case .is, .isNot, .and, .or: return .null
    }
  }
}

private func templateTest(_ name: String, _ value: TemplateValue) throws(TemplateFailure) -> Bool {
  switch name {
  case "none": value.isNull
  case "boolean":
    if case .boolean = value { true } else { false }
  case "integer": value.isInteger
  case "float":
    if case .number = value { true } else { false }
  case "number": value.isNumber
  case "string": value.isString
  case "mapping": value.isObject
  case "iterable": value.isIterable
  case "sequence": value.isArray
  case "defined": !value.isNull
  case "true": value.isTruthy
  case "false": !value.isTruthy
  default: throw TemplateFailure("Unknown type for 'is' operator: " + name)
  }
}

// MARK: - Subscripts

private func evaluateSubscript(
  _ base: TemplateExpression,
  _ index: TemplateExpression,
  _ context: TemplateContext
) throws(TemplateFailure) -> TemplateValue {
  let target = try base.evaluate(context)
  guard case .slice(let start, let end, let step) = index.kind else {
    let key = try index.evaluate(context)
    guard !target.isNull else {
      guard let name = base.variableName else {
        throw TemplateFailure("Trying to access property '\(key.repr())' on null!")
      }
      let state = context.value(named: name) == nil ? "not defined" : "null"
      throw TemplateFailure("'\(name)' is \(state)")
    }
    return try target.element(at: key)
  }
  let count = Int64(try target.count())
  let stride = try step?.evaluate(context).integerValue() ?? 1
  guard stride != 0 else {
    throw TemplateFailure("slice step cannot be zero")
  }
  let (lowerBound, upperBound) = stride < 0 ? (Int64(-1), count - 1) : (0, count)
  func resolve(_ bound: TemplateExpression?, default value: Int64) throws(TemplateFailure) -> Int64
  {
    guard let bound else {
      return value
    }
    let index = try bound.evaluate(context).integerValue()
    return index < 0 ? max(index + count, lowerBound) : min(index, upperBound)
  }
  let first = try resolve(start, default: stride < 0 ? upperBound : lowerBound)
  let last = try resolve(end, default: stride < 0 ? lowerBound : upperBound)
  let indices = Swift.stride(from: first, to: last, by: Int(clamping: stride)).map { Int($0) }
  switch target {
  case .string(let string):
    let scalars = Array(string.unicodeScalars)
    var sliced = String.UnicodeScalarView()
    sliced.append(contentsOf: indices.map { scalars[$0] })
    return .string(String(sliced))
  case .array(let array):
    return .array(indices.map { array.elements[$0] })
  default:
    throw TemplateFailure("Subscripting only supported on arrays and strings")
  }
}

// MARK: - Methods

private func callMethod(
  _ method: String,
  on value: TemplateValue,
  _ arguments: TemplateArguments,
  _ context: TemplateContext
) throws(TemplateFailure) -> TemplateValue {
  switch value {
  case .array(let array):
    switch method {
    case "append":
      try arguments.expect("append method", positional: 1...1, named: 0...0)
      array.elements.append(arguments.positional[0])
      return .null
    case "pop":
      try arguments.expect("pop method", positional: 0...1, named: 0...0)
      return try value.pop(arguments.positional.first ?? .null)
    case "insert":
      try arguments.expect("insert method", positional: 2...2, named: 0...0)
      let index = try arguments.positional[0].integerValue()
      guard index >= 0 && index <= array.elements.count else {
        throw TemplateFailure("Index out of range for insert method")
      }
      array.elements.insert(arguments.positional[1], at: Int(index))
      return .null
    default:
      break
    }
  case .object(let object):
    switch method {
    case "items":
      try arguments.expect("items method", positional: 0...0, named: 0...0)
      return .array(object.entries.map { TemplateValue.array([$0.key.value, $0.value]) })
    case "pop":
      try arguments.expect("pop method", positional: 1...1, named: 0...0)
      return try value.pop(arguments.positional[0])
    case "keys":
      try arguments.expect("keys method", positional: 0...0, named: 0...0)
      return .array(try value.keys())
    case "get":
      try arguments.expect("get method", positional: 1...2, named: 0...0)
      let key = arguments.positional[0]
      guard try value.contains(key) else {
        return arguments.positional.count == 2 ? arguments.positional[1] : .null
      }
      return try value.element(at: key)
    default:
      guard let property = object.entries[.string(method)] else {
        break
      }
      guard property.isCallable else {
        throw TemplateFailure("Property '\(method)' is not callable")
      }
      return try property.call(context, arguments)
    }
  case .string(let string):
    if let result = try callStringMethod(method, on: string, arguments) {
      return result
    }
  default:
    break
  }
  throw TemplateFailure("Unknown method: " + method)
}

private func callStringMethod(
  _ method: String,
  on string: String,
  _ arguments: TemplateArguments
) throws(TemplateFailure) -> TemplateValue? {
  switch method {
  case "strip", "lstrip", "rstrip":
    try arguments.expect("\(method) method", positional: 0...1, named: 0...0)
    let characters = try arguments.positional.first?.stringValue() ?? ""
    return .string(
      templateStrip(
        string,
        characters: characters,
        left: method != "rstrip",
        right: method != "lstrip"
      )
    )
  case "split":
    try arguments.expect("split method", positional: 1...1, named: 0...0)
    let parts = try templateSplit(string, separator: try arguments.positional[0].stringValue())
    return .array(parts.map { .string($0) })
  case "capitalize":
    try arguments.expect("capitalize method", positional: 0...0, named: 0...0)
    return .string(templateCapitalized(string))
  case "upper":
    try arguments.expect("upper method", positional: 0...0, named: 0...0)
    return .string(templateUppercased(string))
  case "lower":
    try arguments.expect("lower method", positional: 0...0, named: 0...0)
    return .string(templateLowercased(string))
  case "endswith":
    try arguments.expect("endswith method", positional: 1...1, named: 0...0)
    return .boolean(templateHasSuffix(string, try arguments.positional[0].stringValue()))
  case "startswith":
    try arguments.expect("startswith method", positional: 1...1, named: 0...0)
    return .boolean(templateHasPrefix(string, try arguments.positional[0].stringValue()))
  case "title":
    try arguments.expect("title method", positional: 0...0, named: 0...0)
    return .string(templateTitleCased(string))
  case "replace":
    try arguments.expect("replace method", positional: 2...3, named: 0...0)
    let count = try arguments.positional.count == 3 ? arguments.positional[2].integerValue() : nil
    return .string(
      templateReplace(
        string,
        try arguments.positional[0].stringValue(),
        try arguments.positional[1].stringValue(),
        count: count
      )
    )
  default:
    return nil
  }
}

// MARK: - Filters

private func evaluateFilter(
  _ parts: [TemplateExpression],
  _ context: TemplateContext
) throws(TemplateFailure) -> TemplateValue {
  guard let first = parts.first else {
    return .null
  }
  var result = try first.evaluate(context)
  for part in parts.dropFirst() {
    guard case .call(let callee, let callArguments) = part.kind else {
      result = try part.evaluate(context).call(context, TemplateArguments([result]))
      continue
    }
    let function = try callee.evaluate(context)
    var arguments = try callArguments.evaluate(context)
    arguments.positional.insert(result, at: 0)
    result = try function.call(context, arguments)
  }
  return result
}
