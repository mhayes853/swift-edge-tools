import OrderedCollections

// MARK: - Builtins

func builtinFunction(named name: String) -> TemplateFunction? {
  switch name {
  case "raise_exception":
    return simpleFunction(name, ["message"]) { arguments throws(TemplateFailure) in
      throw TemplateFailure(try arguments.required("message").stringValue())
    }
  case "tojson":
    return { _, arguments throws(TemplateFailure) in try pythonJSON(arguments) }
  case "items":
    return simpleFunction(name, ["object"]) { arguments throws(TemplateFailure) in
      guard let object = arguments["object"] else {
        return .array([])
      }
      guard case .object(let entries) = object else {
        throw TemplateFailure("Can only get item pairs from a mapping")
      }
      return .array(entries.entries.map { TemplateValue.array([$0.key.value, $0.value]) })
    }
  case "last":
    return simpleFunction(name, ["items"]) { arguments throws(TemplateFailure) in
      guard case .array(let items) = try arguments.required("items") else {
        throw TemplateFailure("object is not a list")
      }
      return items.elements.last ?? .null
    }
  case "trim":
    return simpleFunction(name, ["text"]) { arguments throws(TemplateFailure) in
      let text = try arguments.required("text")
      guard !text.isNull else {
        return text
      }
      return .string(templateStrip(try text.stringValue()))
    }
  case "lower", "upper":
    return simpleFunction(name, ["text"]) { arguments throws(TemplateFailure) in
      let text = try arguments.required("text")
      guard !text.isNull else {
        return text
      }
      let string = try text.stringValue()
      return .string(name == "lower" ? templateLowercased(string) : templateUppercased(string))
    }
  case "default":
    return { _, arguments throws(TemplateFailure) in
      try arguments.expect("default", positional: 2...3, named: 0...1)
      let value = arguments.positional[0]
      let defaultValue = arguments.positional[1]
      var isBoolean = false
      if arguments.positional.count == 3 {
        isBoolean = try arguments.positional[2].booleanValue()
      } else if let boolean = arguments.named("boolean"), !boolean.isNull {
        isBoolean = try boolean.booleanValue()
      }
      guard isBoolean else {
        return value.isNull ? defaultValue : value
      }
      return value.isTruthy ? value : defaultValue
    }
  case "e", "escape":
    return simpleFunction(name, ["text"]) { arguments throws(TemplateFailure) in
      .string(templateHTMLEscaped(try arguments.required("text").stringValue()))
    }
  case "joiner":
    return simpleFunction(name, ["sep"]) { arguments throws(TemplateFailure) in
      let separator = try arguments["sep"]?.stringValue() ?? ""
      var isFirst = true
      return .function(
        simpleFunction("", []) { _ in
          defer { isFirst = false }
          return .string(isFirst ? "" : separator)
        }
      )
    }
  case "count", "length":
    return simpleFunction(name, ["items"]) { arguments throws(TemplateFailure) in
      .integer(Int64(try arguments.required("items").count()))
    }
  case "dictsort":
    return simpleFunction(name, ["value"]) { arguments throws(TemplateFailure) in
      guard let value = arguments["value"] else {
        throw TemplateFailure("dictsort expects exactly 1 argument (TODO: fix implementation)")
      }
      var pairs = [TemplateValue]()
      for key in try sortedTemplateValues(try value.keys()) {
        pairs.append(.array([key, try value.element(at: key)]))
      }
      return .array(pairs)
    }
  case "join":
    return simpleFunction(name, ["items", "d"]) { arguments throws(TemplateFailure) in
      let separator = try arguments["d"]?.stringValue() ?? ""
      if let items = arguments["items"] {
        return try joined(items, separator: separator)
      }
      return .function(
        simpleFunction("", ["items"]) { arguments throws(TemplateFailure) in
          let items = try arguments.required("items")
          guard items.isTruthy && items.isArray else {
            throw TemplateFailure("join expects an array for items, got: " + items.repr())
          }
          return try joined(items, separator: separator)
        }
      )
    }
  case "namespace":
    return { _, arguments throws(TemplateFailure) in
      try arguments.expect("namespace", positional: 0...0, named: 0...Int.max)
      let namespace = TemplateObject()
      for (name, value) in arguments.named {
        namespace.entries[.string(name)] = value
      }
      return .object(namespace)
    }
  case "equalto", "==":
    return simpleFunction(name, ["expected", "actual"]) { arguments throws(TemplateFailure) in
      .boolean(try arguments.required("actual").isEqual(to: arguments.required("expected")))
    }
  case "safe", "string":
    return simpleFunction(name, ["value"]) { arguments throws(TemplateFailure) in
      .string(try arguments.required("value").displayString)
    }
  case "int":
    return simpleFunction(name, ["value"]) { arguments throws(TemplateFailure) in
      .integer(try arguments.required("value").integerConversion)
    }
  case "list":
    return simpleFunction(name, ["items"]) { arguments throws(TemplateFailure) in
      let items = try arguments.required("items")
      guard items.isArray else {
        throw TemplateFailure("object is not iterable")
      }
      return items
    }
  case "in":
    return simpleFunction(name, ["item", "items"]) { arguments throws(TemplateFailure) in
      .boolean(try templateIn(arguments.required("item"), arguments.required("items")))
    }
  case "unique":
    return simpleFunction(name, ["items"]) { arguments throws(TemplateFailure) in
      guard case .array(let items) = try arguments.required("items") else {
        throw TemplateFailure("object is not iterable")
      }
      var seen = Set<TemplatePrimitive>()
      var unique = [TemplateValue]()
      for item in items.elements {
        guard let key = item.primitive else {
          throw TemplateFailure("Unsupported type for hashing: " + item.repr())
        }
        if seen.insert(key).inserted {
          unique.append(item)
        }
      }
      return .array(unique)
    }
  case "select", "reject":
    return { context, arguments throws(TemplateFailure) in
      try arguments.expect(name, positional: 2...Int.max, named: 0...0)
      let items = arguments.positional[0]
      guard !items.isNull else {
        return .array([])
      }
      guard case .array(let array) = items else {
        throw TemplateFailure("object is not iterable: " + items.repr())
      }
      let test = context.value(for: arguments.positional[1])
      guard !test.isNull else {
        throw TemplateFailure("Undefined filter: " + arguments.positional[1].repr())
      }
      let testArguments = Array(arguments.positional.dropFirst(2))
      var selected = [TemplateValue]()
      for item in array.elements {
        let result = try test.call(context, TemplateArguments([item] + testArguments))
        if result.isTruthy == (name == "select") {
          selected.append(item)
        }
      }
      return .array(selected)
    }
  case "map":
    return { context, arguments throws(TemplateFailure) in try mapped(arguments, context) }
  case "indent":
    return simpleFunction(name, ["text", "indent", "first"]) { arguments throws(TemplateFailure) in
      .string(
        templateIndented(
          try arguments.required("text").stringValue(),
          width: try arguments["indent"]?.integerValue() ?? 0,
          indentsFirstLine: try arguments["first"]?.booleanValue() ?? false
        )
      )
    }
  case "selectattr", "rejectattr":
    return { context, arguments throws(TemplateFailure) in
      try selectedAttributes(arguments, context, isSelecting: name == "selectattr")
    }
  case "range":
    return { _, arguments throws(TemplateFailure) in try range(arguments) }
  case "min", "max":
    return { _, arguments throws(TemplateFailure) in
      try extremum(arguments, name: name, isMinimum: name == "min")
    }
  default:
    return nil
  }
}

// MARK: - Simple Functions

private struct SimpleArguments {
  let function: String
  let parameters: [String]
  let values: [TemplateValue?]

  subscript(_ parameter: String) -> TemplateValue? {
    self.parameters.firstIndex(of: parameter).flatMap { self.values[$0] }
  }

  func required(_ parameter: String) throws(TemplateFailure) -> TemplateValue {
    guard let value = self[parameter] else {
      throw TemplateFailure("Missing argument \(parameter) for function \(self.function)")
    }
    return value
  }
}

private func simpleFunction(
  _ name: String,
  _ parameters: [String],
  _ body: @escaping (SimpleArguments) throws(TemplateFailure) -> TemplateValue
) -> TemplateFunction {
  { _, arguments throws(TemplateFailure) in
    var values = [TemplateValue?](repeating: nil, count: parameters.count)
    for (index, argument) in arguments.positional.enumerated() {
      guard index < parameters.count else {
        throw TemplateFailure("Too many positional params for " + name)
      }
      values[index] = argument
    }
    for (argumentName, value) in arguments.named {
      guard let index = parameters.lastIndex(of: argumentName) else {
        throw TemplateFailure("Unknown argument \(argumentName) for function \(name)")
      }
      values[index] = value
    }
    return try body(SimpleArguments(function: name, parameters: parameters, values: values))
  }
}

// MARK: - Collections

private func joined(_ items: TemplateValue, separator: String) throws(TemplateFailure)
  -> TemplateValue
{
  guard case .array(let array) = items else {
    throw TemplateFailure("object is not iterable: " + items.repr())
  }
  return .string(array.elements.map { $0.displayString }.joined(separator: separator))
}

private func sortedTemplateValues(_ values: [TemplateValue]) throws(TemplateFailure)
  -> [TemplateValue]
{
  for (lhs, rhs) in zip(values, values.dropFirst()) {
    _ = try lhs.isLess(than: rhs)
  }
  return values.sorted { (try? $0.isLess(than: $1)) ?? false }
}

private func mapped(
  _ arguments: TemplateArguments,
  _ context: TemplateContext
) throws(TemplateFailure) -> TemplateValue {
  let namedCount = arguments.named.count
  let mapsAttribute =
    arguments.named("attribute") != nil && namedCount == 1
    || arguments.named("default") != nil && namedCount == 2
  guard let items = arguments.positional.first else {
    throw TemplateFailure("Invalid or unsupported arguments for map")
  }
  guard case .array(let array) = items else {
    throw TemplateFailure("Value is not an array or object: " + items.repr())
  }
  var values = [TemplateValue]()
  if arguments.positional.count == 1 && mapsAttribute {
    let attribute = arguments.named("attribute") ?? .null
    let defaultValue = arguments.named("default") ?? .null
    for item in array.elements {
      let attributeValue = try item.element(at: attribute)
      values.append(attributeValue.isNull ? defaultValue : attributeValue)
    }
    return .array(values)
  }
  guard arguments.named.isEmpty && arguments.positional.count >= 2 else {
    throw TemplateFailure("Invalid or unsupported arguments for map")
  }
  let function = context.value(for: arguments.positional[1])
  guard !function.isNull else {
    throw TemplateFailure("Undefined filter: " + arguments.positional[1].repr())
  }
  let extraArguments = Array(arguments.positional.dropFirst(2))
  for item in array.elements {
    values.append(try function.call(context, TemplateArguments([item] + extraArguments)))
  }
  return .array(values)
}

private func selectedAttributes(
  _ arguments: TemplateArguments,
  _ context: TemplateContext,
  isSelecting: Bool
) throws(TemplateFailure) -> TemplateValue {
  try arguments.expect(
    isSelecting ? "selectattr" : "rejectattr",
    positional: 2...Int.max,
    named: 0...0
  )
  let items = arguments.positional[0]
  guard !items.isNull else {
    return .array([])
  }
  guard case .array(let array) = items else {
    throw TemplateFailure("object is not iterable: " + items.repr())
  }
  let attribute = TemplateValue.string(try arguments.positional[1].stringValue())
  var test: TemplateValue?
  if arguments.positional.count >= 3 {
    let function = context.value(for: arguments.positional[2])
    guard !function.isNull else {
      throw TemplateFailure("Undefined test: " + arguments.positional[2].repr())
    }
    test = function
  }
  var selected = [TemplateValue]()
  for item in array.elements {
    let attributeValue = try item.element(at: attribute)
    guard let test else {
      selected.append(attributeValue)
      continue
    }
    let testArguments = TemplateArguments(
      [attributeValue] + arguments.positional.dropFirst(3),
      named: arguments.named
    )
    if try test.call(context, testArguments).isTruthy == isSelecting {
      selected.append(item)
    }
  }
  return .array(selected)
}

private func range(_ arguments: TemplateArguments) throws(TemplateFailure) -> TemplateValue {
  guard arguments.positional.count <= 3 else {
    throw TemplateFailure("range expects at most 3 positional arguments")
  }
  var bounds: [Int64?] = [nil, nil, nil]
  if arguments.positional.count == 1 {
    bounds[1] = try arguments.positional[0].integerValue()
  } else {
    for (index, argument) in arguments.positional.enumerated() {
      bounds[index] = try argument.integerValue()
    }
  }
  for (name, value) in arguments.named {
    guard let index = ["start", "end", "step"].firstIndex(of: name) else {
      throw TemplateFailure("Unknown argument \(name) for function range")
    }
    guard bounds[index] == nil else {
      throw TemplateFailure("Duplicate argument \(name) for function range")
    }
    bounds[index] = try value.integerValue()
  }
  guard let end = bounds[1] else {
    throw TemplateFailure("Missing required argument 'end' for function range")
  }
  let step = bounds[2] ?? 1
  guard step != 0 else {
    throw TemplateFailure("range() arg 3 must not be zero")
  }
  return .array(
    stride(from: bounds[0] ?? 0, to: end, by: Int(clamping: step)).map { .integer($0) }
  )
}

private func extremum(
  _ arguments: TemplateArguments,
  name: String,
  isMinimum: Bool
) throws(TemplateFailure) -> TemplateValue {
  try arguments.expect(name, positional: 1...1, named: 0...0)
  guard case .array(let sequence) = arguments.positional[0] else {
    throw TemplateFailure("\(name) expects a sequence")
  }
  return sequence.elements.dropFirst()
    .reduce(sequence.elements.first ?? .null) { extremum, element in
      let isLess =
        if element.isNumber && extremum.isNumber {
          (try? element.doubleValue() < extremum.doubleValue()) ?? false
        } else {
          templateIsLess(element.displayString, extremum.displayString)
        }
      return isLess == isMinimum ? element : extremum
    }
}

// MARK: - JSON

private func pythonJSON(_ arguments: TemplateArguments) throws(TemplateFailure) -> TemplateValue {
  guard let value = arguments.positional.first else {
    throw TemplateFailure("tojson expects a value to serialize")
  }
  guard arguments.positional.count <= 2 else {
    throw TemplateFailure("tojson expects at most two positional arguments")
  }
  var options = PythonJSONOptions()
  if arguments.positional.count > 1 {
    options.indent = try arguments.positional[1].integerValue()
  }
  for (name, option) in arguments.named {
    switch name {
    case "indent": options.indent = try option.isNull ? -1 : option.integerValue()
    case "ensure_ascii": options.ensuresASCII = try option.booleanValue()
    case "sort_keys": options.sortsKeys = try option.booleanValue()
    default: throw TemplateFailure("Unknown argument \(name) for function tojson")
    }
  }
  var output = ""
  appendPythonJSON(value, to: &output, options: options, level: 0)
  return .string(output)
}

private struct PythonJSONOptions {
  var ensuresASCII = false
  var indent: Int64 = -1
  var sortsKeys = false
}

private func appendPythonJSON(
  _ value: TemplateValue,
  to output: inout String,
  options: PythonJSONOptions,
  level: Int
) {
  func appendBreak(_ level: Int) {
    if options.indent >= 0 {
      output += "\n" + String(repeating: " ", count: level * Int(clamping: options.indent))
    }
  }
  func appendSeparator(_ index: Int) {
    guard index > 0 else {
      return
    }
    output += ","
    if options.indent < 0 {
      output += " "
    } else {
      appendBreak(level + 1)
    }
  }
  switch value {
  case .null:
    output += "null"
  case .boolean(let value):
    output += value ? "true" : "false"
  case .integer(let value):
    output += "\(value)"
  case .number(let value):
    output += templateJSONNumber(value)
  case .string(let value):
    appendJSONString(value, ensuringASCII: options.ensuresASCII, to: &output)
  case .array(let array) where array.elements.isEmpty:
    output += "[]"
  case .array(let array):
    output += "["
    appendBreak(level + 1)
    for (index, element) in array.elements.enumerated() {
      appendSeparator(index)
      appendPythonJSON(element, to: &output, options: options, level: level + 1)
    }
    appendBreak(level)
    output += "]"
  case .object(let object):
    let entries = object.jsonEntries
    guard !entries.isEmpty else {
      output += "{}"
      return
    }
    let keys = options.sortsKeys ? entries.keys.sorted(by: templateIsLess) : Array(entries.keys)
    output += "{"
    appendBreak(level + 1)
    for (index, key) in keys.enumerated() {
      appendSeparator(index)
      appendJSONString(key, ensuringASCII: options.ensuresASCII, to: &output)
      output += ": "
      appendPythonJSON(entries[key] ?? .null, to: &output, options: options, level: level + 1)
    }
    appendBreak(level)
    output += "}"
  }
}
