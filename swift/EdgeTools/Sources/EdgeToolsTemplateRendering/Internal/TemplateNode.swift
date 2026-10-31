import OrderedCollections

// MARK: - TemplateNode

struct TemplateNode: Sendable {
  let position: Int
  let kind: Kind

  indirect enum Kind: Sendable {
    case sequence([TemplateNode])
    case text(String)
    case expression(TemplateExpression)
    case conditional([(condition: TemplateExpression?, body: TemplateNode)])
    case loopControl(TemplateLoopControl)
    case loop(TemplateLoop)
    case macro(TemplateMacro)
    case filter(TemplateExpression, body: TemplateNode)
    case set(namespace: String?, names: [String], value: TemplateExpression)
    case setBlock(name: String, body: TemplateNode)
    case call(TemplateExpression, body: TemplateNode)
  }

  init(_ kind: Kind, at position: Int) {
    self.position = position
    self.kind = kind
  }
}

enum TemplateLoopControl: Sendable {
  case `break`
  case `continue`
}

struct TemplateLoop: Sendable {
  let names: [String]
  let iterable: TemplateExpression
  let condition: TemplateExpression?
  let body: TemplateNode
  let isRecursive: Bool
  let elseBody: TemplateNode?
}

struct TemplateMacro: Sendable {
  struct Parameter: Sendable {
    let name: String
    let defaultValue: TemplateExpression?
  }

  let name: String
  let parameters: [Parameter]
  let body: TemplateNode
  let parameterIndices: [String: Int]

  init(name: String, parameters: [Parameter], body: TemplateNode) {
    self.name = name
    self.parameters = parameters
    self.body = body
    self.parameterIndices = parameters.indices.reduce(into: [:]) { indices, index in
      if !parameters[index].name.isEmpty {
        indices[parameters[index].name] = index
      }
    }
  }
}

// MARK: - TemplateLoopSignal

struct TemplateLoopSignal {
  let control: TemplateLoopControl
  let position: Int

  var failure: TemplateFailure {
    let keyword = self.control == .break ? "break" : "continue"
    return TemplateFailure("\(keyword) outside of a loop", position: self.position)
  }
}

// MARK: - Rendering

extension TemplateNode {
  func render(_ context: TemplateContext) throws(TemplateFailure) -> String {
    var output = ""
    if let signal = try self.render(into: &output, context) {
      throw signal.failure
    }
    return output
  }

  func render(
    into output: inout String,
    _ context: TemplateContext
  ) throws(TemplateFailure) -> TemplateLoopSignal? {
    do throws(TemplateFailure) {
      return try self.renderKind(into: &output, context)
    } catch {
      throw error.located(at: self.position)
    }
  }

  private func renderKind(
    into output: inout String,
    _ context: TemplateContext
  ) throws(TemplateFailure) -> TemplateLoopSignal? {
    switch self.kind {
    case .sequence(let children):
      for child in children {
        if let signal = try child.render(into: &output, context) {
          return signal
        }
      }
    case .text(let text):
      output += text
    case .expression(let expression):
      switch try expression.evaluate(context) {
      case .string(let value): output += value
      case .boolean(let value): output += value ? "True" : "False"
      case .null: break
      case let value: output += value.repr()
      }
    case .conditional(let branches):
      for branch in branches {
        if try branch.condition?.evaluate(context).isTruthy ?? true {
          return try branch.body.render(into: &output, context)
        }
      }
    case .loopControl(let control):
      return TemplateLoopSignal(control: control, position: self.position)
    case .loop(let loop):
      return try loop.render(into: &output, context)
    case .macro(let macro):
      context.set(macro.name, .function(macro.function(definedIn: context)))
    case .filter(let filter, let body):
      let function = try filter.evaluate(context)
      guard function.isCallable else {
        throw TemplateFailure("Filter must be a callable: " + function.repr())
      }
      var rendered = ""
      if let signal = try body.render(into: &rendered, context) {
        return signal
      }
      output += try function.call(context, TemplateArguments([.string(rendered)])).displayString
    case .set(let namespace?, let names, let value):
      let object = context.value(named: namespace) ?? .null
      guard object.isObject else {
        throw TemplateFailure("Namespace '\(namespace)' is not an object")
      }
      try object.setElement(try value.evaluate(context), at: .string(names[0]))
    case .set(nil, let names, let value):
      try templateAssign(names, try value.evaluate(context), in: context)
    case .setBlock(let name, let body):
      var rendered = ""
      if let signal = try body.render(into: &rendered, context) {
        return signal
      }
      context.set(name, .string(rendered))
    case .call(let expression, let body):
      context.set(
        "caller",
        .function { _, _ throws(TemplateFailure) in .string(try body.render(context)) }
      )
      guard case .call(let callee, let callArguments) = expression.kind else {
        throw TemplateFailure("Invalid call block syntax - expected function call")
      }
      let function = try callee.evaluate(context)
      guard function.isCallable else {
        throw TemplateFailure("Call target must be callable: " + function.repr())
      }
      let arguments = try callArguments.evaluate(context)
      output += try function.call(context, arguments).displayString
    }
    return nil
  }
}

// MARK: - Loops

extension TemplateLoop {
  fileprivate func render(
    into output: inout String,
    _ context: TemplateContext
  ) throws(TemplateFailure) -> TemplateLoopSignal? {
    func visit(
      _ iterable: TemplateValue,
      into output: inout String
    ) throws(TemplateFailure) -> TemplateLoopSignal? {
      var items = [TemplateValue]()
      if !iterable.isNull {
        guard let elements = iterable.iterationElements else {
          throw TemplateFailure("For loop iterable must be iterable: " + iterable.repr())
        }
        for item in elements {
          try templateAssign(self.names, item, in: context)
          if try self.condition?.evaluate(context).isTruthy ?? true {
            items.append(item)
          }
        }
      }
      guard !items.isEmpty else {
        return try self.elseBody?.render(into: &output, context)
      }
      let loop = TemplateObject(function: self.isRecursive ? callLoop : nil)
      loop.entries[.string("length")] = .integer(Int64(items.count))
      var cycleIndex = 0
      let cycle = TemplateValue.function { _, arguments throws(TemplateFailure) in
        guard !arguments.positional.isEmpty && arguments.named.isEmpty else {
          throw TemplateFailure(
            "cycle() expects at least 1 positional argument and no named arg"
          )
        }
        let item = arguments.positional[cycleIndex % arguments.positional.count]
        cycleIndex = (cycleIndex + 1) % arguments.positional.count
        return item
      }
      loop.entries[.string("cycle")] = cycle
      let loopContext = TemplateContext(parent: context)
      loopContext.set("loop", .object(loop))
      for (index, item) in items.enumerated() {
        try templateAssign(self.names, item, in: loopContext)
        let count = Int64(items.count)
        let properties: KeyValuePairs<String, TemplateValue> = [
          "index": .integer(Int64(index) + 1),
          "index0": .integer(Int64(index)),
          "revindex": .integer(count - Int64(index)),
          "revindex0": .integer(count - Int64(index) - 1),
          "length": .integer(count),
          "first": .boolean(index == 0),
          "last": .boolean(index == items.count - 1),
          "previtem": index > 0 ? items[index - 1] : .null,
          "nextitem": index < items.count - 1 ? items[index + 1] : .null
        ]
        for (name, value) in properties {
          loop.entries[.string(name)] = value
        }
        if try self.body.render(into: &output, loopContext)?.control == .break {
          break
        }
      }
      return nil
    }

    func callLoop(
      _ callContext: TemplateContext,
      _ arguments: TemplateArguments
    ) throws(TemplateFailure) -> TemplateValue {
      guard arguments.positional.count == 1, arguments.named.isEmpty,
        arguments.positional[0].isArray
      else {
        throw TemplateFailure("loop() expects exactly 1 positional iterable argument")
      }
      var rendered = ""
      if let signal = try visit(arguments.positional[0], into: &rendered) {
        throw signal.failure
      }
      return .string(rendered)
    }

    return try visit(try self.iterable.evaluate(context), into: &output)
  }
}

func templateAssign(
  _ names: [String],
  _ value: TemplateValue,
  in context: TemplateContext
) throws(TemplateFailure) {
  guard names.count != 1 else {
    context.set(names[0], value)
    return
  }
  guard case .array(let array) = value, array.elements.count == names.count else {
    throw TemplateFailure(
      "Mismatched number of variables and items in destructuring assignment"
    )
  }
  for (name, element) in zip(names, array.elements) {
    context.set(name, element)
  }
}

// MARK: - Macros

extension TemplateMacro {
  fileprivate func function(definedIn definitionContext: TemplateContext) -> TemplateFunction {
    { callContext, arguments throws(TemplateFailure) in
      let executionContext = TemplateContext(parent: definitionContext)
      if let caller = callContext.value(named: "caller") {
        executionContext.set("caller", caller)
      }
      var isProvided = [Bool](repeating: false, count: self.parameters.count)
      for (index, argument) in arguments.positional.enumerated() {
        guard index < self.parameters.count else {
          throw TemplateFailure("Too many positional arguments for macro " + self.name)
        }
        isProvided[index] = true
        executionContext.set(self.parameters[index].name, argument)
      }
      for (name, value) in arguments.named {
        guard let index = self.parameterIndices[name] else {
          throw TemplateFailure("Unknown parameter name for macro \(self.name): \(name)")
        }
        executionContext.set(name, value)
        isProvided[index] = true
      }
      for (index, parameter) in self.parameters.enumerated() where !isProvided[index] {
        if let defaultValue = parameter.defaultValue {
          executionContext.set(parameter.name, try defaultValue.evaluate(callContext))
        }
      }
      return .string(try self.body.render(executionContext))
    }
  }
}
