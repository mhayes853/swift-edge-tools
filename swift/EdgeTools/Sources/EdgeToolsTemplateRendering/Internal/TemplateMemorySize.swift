// MARK: - Memory Size

// Most of a parsed template is per-allocation overhead from its boxed nodes, which scales with
// the pointer size. The coefficients were fitted against measured heap growth for real chat
// templates on 64-bit Linux, where the estimate lands within about 20% for templates of at
// least 1 KB.
func templateMemorySize(sourceSize: Int, root: TemplateNode) -> Int {
  var counts = SyntaxCounts()
  counts.add(root)
  let word = MemoryLayout<Int>.size
  return sourceSize * 11 / 4 + counts.nodes * 12 * word + counts.expressions * 6 * word
}

// MARK: - SyntaxCounts

private struct SyntaxCounts {
  var nodes = 0
  var expressions = 0

  mutating func add(_ node: TemplateNode) {
    self.nodes += 1
    switch node.kind {
    case .sequence(let children):
      for child in children {
        self.add(child)
      }
    case .text, .loopControl:
      break
    case .expression(let expression), .set(_, _, let expression):
      self.add(expression)
    case .conditional(let branches):
      for branch in branches {
        self.add(branch.condition)
        self.add(branch.body)
      }
    case .loop(let loop):
      self.add(loop.iterable)
      self.add(loop.condition)
      self.add(loop.body)
      if let elseBody = loop.elseBody {
        self.add(elseBody)
      }
    case .macro(let macro):
      for parameter in macro.parameters {
        self.add(parameter.defaultValue)
      }
      self.add(macro.body)
    case .filter(let expression, let body), .call(let expression, let body):
      self.add(expression)
      self.add(body)
    case .setBlock(_, let body):
      self.add(body)
    }
  }

  mutating func add(_ expression: TemplateExpression?) {
    guard let expression else {
      return
    }
    self.expressions += 1
    switch expression.kind {
    case .variable, .literal:
      break
    case .array(let elements), .filter(let elements):
      for element in elements {
        self.add(element)
      }
    case .dictionary(let entries):
      for entry in entries {
        self.add(entry.key)
        self.add(entry.value)
      }
    case .subscripted(let base, .key(let key)):
      self.add(base)
      self.add(key)
    case .subscripted(let base, .slice(let start, let end, let step)):
      self.add(base)
      self.add(start)
      self.add(end)
      self.add(step)
    case .unary(_, let operand):
      self.add(operand)
    case .binary(_, let left, let right):
      self.add(left)
      self.add(right)
    case .conditional(let condition, let then, let otherwise):
      self.add(condition)
      self.add(then)
      self.add(otherwise)
    case .methodCall(let base, _, let arguments), .call(let base, let arguments):
      self.add(base)
      for argument in arguments.positional {
        self.add(argument)
      }
      for argument in arguments.named {
        self.add(argument.value)
      }
    }
  }
}
