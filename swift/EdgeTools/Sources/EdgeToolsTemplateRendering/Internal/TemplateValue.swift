import EdgeToolsCore
import OrderedCollections

// MARK: - TemplatePrimitive

enum TemplatePrimitive: Hashable, Sendable {
  case null
  case boolean(Bool)
  case integer(Int64)
  case number(Double)
  case string(String)

  var value: TemplateValue {
    switch self {
    case .null: .null
    case .boolean(let value): .boolean(value)
    case .integer(let value): .integer(value)
    case .number(let value): .number(value)
    case .string(let value): .string(value)
    }
  }

  var jsonKey: String {
    switch self {
    case .string(let value): value
    case .null: "null"
    case .boolean(let value): value ? "true" : "false"
    case .integer(let value): "\(value)"
    case .number(let value): templateJSONNumber(value)
    }
  }

  static func == (lhs: Self, rhs: Self) -> Bool {
    switch (lhs, rhs) {
    case (.null, .null): true
    case (.boolean(let lhs), .boolean(let rhs)): lhs == rhs
    case (.integer(let lhs), .integer(let rhs)): lhs == rhs
    case (.number(let lhs), .number(let rhs)): lhs == rhs
    case (.integer(let lhs), .number(let rhs)), (.number(let rhs), .integer(let lhs)):
      Double(lhs) == rhs
    case (.string(let lhs), .string(let rhs)): lhs == rhs
    default: false
    }
  }

  func hash(into hasher: inout Hasher) {
    switch self {
    case .null:
      hasher.combine(0)
    case .boolean(let value):
      hasher.combine(1)
      hasher.combine(value)
    case .integer(let value):
      hasher.combine(2)
      hasher.combine(Double(value))
    case .number(let value):
      hasher.combine(2)
      hasher.combine(value)
    case .string(let value):
      hasher.combine(3)
      hasher.combine(value)
    }
  }
}

// MARK: - TemplateValue

enum TemplateValue {
  case null
  case boolean(Bool)
  case integer(Int64)
  case number(Double)
  case string(String)
  case array(TemplateArray)
  case object(TemplateObject)
}

final class TemplateArray {
  var elements: [TemplateValue]

  init(_ elements: [TemplateValue] = []) {
    self.elements = elements
  }
}

final class TemplateObject {
  var entries: OrderedDictionary<TemplatePrimitive, TemplateValue>
  let function: TemplateFunction?

  init(
    _ entries: OrderedDictionary<TemplatePrimitive, TemplateValue> = [:],
    function: TemplateFunction? = nil
  ) {
    self.entries = entries
    self.function = function
  }
}

typealias TemplateFunction =
  (TemplateContext, TemplateArguments) throws(TemplateFailure) -> TemplateValue

struct TemplateArguments {
  var positional: [TemplateValue]
  var named: [(name: String, value: TemplateValue)]

  init(_ positional: [TemplateValue] = [], named: [(name: String, value: TemplateValue)] = []) {
    self.positional = positional
    self.named = named
  }

  func named(_ name: String) -> TemplateValue? {
    self.named.first { $0.name == name }?.value
  }

  func expect(
    _ name: String,
    positional: ClosedRange<Int>,
    named: ClosedRange<Int>
  ) throws(TemplateFailure) {
    guard positional.contains(self.positional.count) && named.contains(self.named.count) else {
      throw TemplateFailure(
        "\(name) must have between \(positional.lowerBound) and \(positional.upperBound) "
          + "positional arguments and between \(named.lowerBound) and \(named.upperBound) "
          + "keyword arguments"
      )
    }
  }
}

// MARK: - Construction

extension TemplateValue {
  init(_ value: EdgeToolsValue) {
    switch value {
    case .null: self = .null
    case .boolean(let value): self = .boolean(value)
    case .integer(let value): self = .integer(Int64(value))
    case .number(let value): self = value.isFinite ? .number(value) : .null
    case .string(let value): self = .string(value)
    case .array(let values): self = .array(values.map(TemplateValue.init))
    case .object(let object):
      self = .object(
        TemplateObject(
          OrderedDictionary(
            uniqueKeysWithValues: object.lazy.map { (.string($0.key), TemplateValue($0.value)) }
          )
        )
      )
    }
  }

  static func array(_ elements: [TemplateValue]) -> Self {
    .array(TemplateArray(elements))
  }

  static func function(_ function: @escaping TemplateFunction) -> Self {
    .object(TemplateObject(function: function))
  }

  var edgeToolsValue: EdgeToolsValue {
    switch self {
    case .null: .null
    case .boolean(let value): .boolean(value)
    case .integer(let value): Int(exactly: value).map { .integer($0) } ?? .number(Double(value))
    case .number(let value): .number(value)
    case .string(let value): .string(value)
    case .array(let array): .array(array.elements.map { $0.edgeToolsValue })
    case .object(let object): .object(object.jsonEntries.mapValues { $0.edgeToolsValue })
    }
  }
}

extension TemplateObject {
  var jsonEntries: OrderedDictionary<String, TemplateValue> {
    var entries = OrderedDictionary<String, TemplateValue>()
    for (key, value) in self.entries {
      entries[key.jsonKey] = value
    }
    if self.function != nil {
      entries["__callable__"] = .boolean(true)
    }
    return entries
  }
}

// MARK: - Inspection

extension TemplateValue {
  var primitive: TemplatePrimitive? {
    switch self {
    case .null: .null
    case .boolean(let value): .boolean(value)
    case .integer(let value): .integer(value)
    case .number(let value): .number(value)
    case .string(let value): .string(value)
    case .array, .object: nil
    }
  }

  var isNull: Bool {
    if case .null = self {
      return true
    }
    return false
  }

  var isString: Bool {
    if case .string = self {
      return true
    }
    return false
  }

  var isNumber: Bool {
    switch self {
    case .integer, .number: true
    default: false
    }
  }

  var isArray: Bool {
    if case .array = self {
      return true
    }
    return false
  }

  var isObject: Bool {
    if case .object = self {
      return true
    }
    return false
  }

  var isCallable: Bool {
    if case .object(let object) = self {
      return object.function != nil
    }
    return false
  }

  var isIterable: Bool {
    switch self {
    case .array, .object, .string: true
    default: false
    }
  }

  var isTruthy: Bool {
    switch self {
    case .null: false
    case .boolean(let value): value
    case .integer(let value): value != 0
    case .number(let value): value != 0
    case .string(let value): !value.isEmpty
    case .array(let array): !array.elements.isEmpty
    case .object: true
    }
  }

  func count() throws(TemplateFailure) -> Int {
    switch self {
    case .object(let object): object.entries.count
    case .array(let array): array.elements.count
    case .string(let value): value.unicodeScalars.count
    default: throw TemplateFailure("Value is not an array or object: " + self.repr())
    }
  }

  func keys() throws(TemplateFailure) -> [TemplateValue] {
    guard case .object(let object) = self else {
      throw TemplateFailure("Value is not an object: " + self.repr())
    }
    return object.entries.keys.map { $0.value }
  }

  var iterationElements: [TemplateValue]? {
    switch self {
    case .array(let array): array.elements
    case .object(let object): object.entries.keys.map { $0.value }
    case .string(let value): value.unicodeScalars.map { .string(String($0)) }
    default: nil
    }
  }
}

// MARK: - Primitive Access

extension TemplateValue {
  func integerValue() throws(TemplateFailure) -> Int64 {
    switch self {
    case .integer(let value): value
    case .number(let value): templateTruncatedInteger(value)
    default: throw self.typeMismatch(expecting: "number")
    }
  }

  func doubleValue() throws(TemplateFailure) -> Double {
    switch self {
    case .integer(let value): Double(value)
    case .number(let value): value
    default: throw self.typeMismatch(expecting: "number")
    }
  }

  func stringValue() throws(TemplateFailure) -> String {
    guard case .string(let value) = self else {
      throw self.typeMismatch(expecting: "string")
    }
    return value
  }

  func booleanValue() throws(TemplateFailure) -> Bool {
    guard case .boolean(let value) = self else {
      throw self.typeMismatch(expecting: "boolean")
    }
    return value
  }

  private func typeMismatch(expecting type: String) -> TemplateFailure {
    let name =
      switch self {
      case .null: "null"
      case .boolean: "boolean"
      case .integer, .number: "number"
      case .string: "string"
      case .array, .object: ""
      }
    guard !name.isEmpty else {
      return TemplateFailure("get<T> not defined for this value type: " + self.repr())
    }
    return TemplateFailure("type must be \(type), but is \(name)")
  }
}

// MARK: - Python Conversions

extension TemplateValue {
  var displayString: String {
    switch self {
    case .string(let value): value
    case .integer(let value): "\(value)"
    case .number(let value): templateFixedNotation(value)
    case .boolean(let value): value ? "True" : "False"
    case .null: "None"
    case .array, .object: self.repr()
    }
  }

  var integerConversion: Int64 {
    switch self {
    case .null: 0
    case .boolean(let value): value ? 1 : 0
    case .integer(let value): value
    case .number(let value): templateTruncatedInteger(value)
    case .string(let value): templateLeadingInteger(value)
    case .array, .object: 0
    }
  }

  func repr() -> String {
    var output = ""
    self.appendRepr(to: &output)
    return output
  }

  private func appendRepr(to output: inout String) {
    switch self {
    case .null:
      output += "null"
    case .boolean(let value):
      output += value ? "True" : "False"
    case .integer(let value):
      output += "\(value)"
    case .number(let value):
      output += templateJSONNumber(value)
    case .string(let value):
      appendPythonString(value, to: &output)
    case .array(let array):
      output += "["
      for (index, element) in array.elements.enumerated() {
        output += index > 0 ? ", " : ""
        element.appendRepr(to: &output)
      }
      output += "]"
    case .object(let object):
      output += "{"
      for (index, (key, value)) in object.entries.enumerated() {
        output += index > 0 ? ", " : ""
        if case .string(let key) = key {
          appendPythonString(key, to: &output)
        } else {
          output += "'" + key.jsonKey + "'"
        }
        output += ": "
        value.appendRepr(to: &output)
      }
      output += "}"
    }
  }
}

// MARK: - Comparison

extension TemplateValue {
  func isEqual(to other: TemplateValue) -> Bool {
    if self.isCallable || other.isCallable {
      guard case .object(let lhs) = self, case .object(let rhs) = other, lhs === rhs else {
        return false
      }
    }
    switch self {
    case .array(let lhs):
      guard case .array(let rhs) = other, lhs.elements.count == rhs.elements.count else {
        return false
      }
      return zip(lhs.elements, rhs.elements)
        .allSatisfy { lhs, rhs in
          lhs.isTruthy && rhs.isTruthy && lhs.isEqual(to: rhs)
        }
    case .object(let lhs):
      guard case .object(let rhs) = other, lhs.entries.count == rhs.entries.count else {
        return false
      }
      return lhs.entries.allSatisfy { key, value in
        guard value.isTruthy, let other = rhs.entries[key] else {
          return false
        }
        return value.isEqual(to: other)
      }
    default:
      return self.primitive == (other.primitive ?? .null)
    }
  }

  func isLess(than other: TemplateValue) throws(TemplateFailure) -> Bool {
    try self.compare(to: other, symbol: "<", ifNumbers: <, ifStrings: templateIsLess)
  }

  func isGreater(than other: TemplateValue) throws(TemplateFailure) -> Bool {
    try self.compare(to: other, symbol: ">", ifNumbers: >) { templateIsLess($1, $0) }
  }

  func contains(_ value: TemplateValue) throws(TemplateFailure) -> Bool {
    switch self {
    case .null:
      throw TemplateFailure("Undefined value or reference")
    case .array(let array):
      return array.elements.contains { $0.isTruthy && $0.isEqual(to: value) }
    case .object(let object):
      return object.entries[try value.hashableKey()] != nil
    default:
      throw TemplateFailure(
        "contains can only be called on arrays and objects: " + self.repr()
      )
    }
  }

  func hashableKey() throws(TemplateFailure) -> TemplatePrimitive {
    guard let primitive else {
      throw TemplateFailure("Unhashable type: " + self.repr())
    }
    return primitive
  }

  private func compare(
    to other: TemplateValue,
    symbol: String,
    ifNumbers: (Double, Double) -> Bool,
    ifStrings: (String, String) -> Bool
  ) throws(TemplateFailure) -> Bool {
    switch (self, other) {
    case (.null, _):
      throw TemplateFailure("Undefined value or reference")
    case (.string(let lhs), .string(let rhs)):
      return ifStrings(lhs, rhs)
    case _ where self.isNumber && other.isNumber:
      return ifNumbers(try self.doubleValue(), try other.doubleValue())
    default:
      throw TemplateFailure(
        "Cannot compare values: \(self.repr()) \(symbol) \(other.repr())"
      )
    }
  }
}

func templateIn(_ value: TemplateValue, _ container: TemplateValue) throws(TemplateFailure) -> Bool
{
  if container.isArray || container.isObject {
    return try container.contains(value)
  }
  guard case .string(let substring) = value, case .string(let string) = container else {
    return false
  }
  return templateContains(string, substring)
}

// MARK: - Element Access

extension TemplateValue {
  func element(at key: TemplateValue) throws(TemplateFailure) -> TemplateValue {
    switch self {
    case .array(let array):
      guard case .integer(let index) = key else {
        return .null
      }
      let resolvedIndex = index < 0 ? index + Int64(array.elements.count) : index
      guard resolvedIndex >= 0 && resolvedIndex < array.elements.count else {
        throw TemplateFailure("list index out of range")
      }
      return array.elements[Int(resolvedIndex)]
    case .object(let object):
      return object.entries[try key.hashableKey()] ?? .null
    default:
      return .null
    }
  }

  func setElement(_ value: TemplateValue, at key: TemplateValue) throws(TemplateFailure) {
    guard case .object(let object) = self else {
      throw TemplateFailure("Value is not an object: " + self.repr())
    }
    object.entries[try key.hashableKey()] = value
  }

  func pop(_ key: TemplateValue) throws(TemplateFailure) -> TemplateValue {
    switch self {
    case .array(let array):
      guard !array.elements.isEmpty else {
        throw TemplateFailure("pop from empty list")
      }
      if key.isNull {
        return array.elements.removeLast()
      }
      guard case .integer(let index) = key else {
        throw TemplateFailure("pop index must be an integer: " + key.repr())
      }
      guard index >= 0 && index < array.elements.count else {
        throw TemplateFailure("pop index out of range: " + key.repr())
      }
      return array.elements.remove(at: Int(index))
    case .object(let object):
      guard let value = object.entries.removeValue(forKey: try key.hashableKey()) else {
        throw TemplateFailure("Key not found: " + key.repr())
      }
      return value
    default:
      throw TemplateFailure("Value is not an array or object: " + self.repr())
    }
  }

  func call(_ context: TemplateContext, _ arguments: TemplateArguments) throws(TemplateFailure)
    -> TemplateValue
  {
    guard case .object(let object) = self, let function = object.function else {
      throw TemplateFailure("Value is not callable: " + self.repr())
    }
    return try function(context, arguments)
  }
}

// MARK: - Arithmetic

extension TemplateValue {
  func adding(_ other: TemplateValue) throws(TemplateFailure) -> TemplateValue {
    switch (self, other) {
    case _ where self.isString || other.isString:
      .string(self.displayString + other.displayString)
    case (.integer(let lhs), .integer(let rhs)):
      .integer(lhs &+ rhs)
    case (.array(let lhs), .array(let rhs)):
      .array(lhs.elements + rhs.elements)
    default:
      .number(try self.doubleValue() + other.doubleValue())
    }
  }

  func subtracting(_ other: TemplateValue) throws(TemplateFailure) -> TemplateValue {
    if case .integer(let lhs) = self, case .integer(let rhs) = other {
      return .integer(lhs &- rhs)
    }
    return .number(try self.doubleValue() - other.doubleValue())
  }

  func multiplying(_ other: TemplateValue) throws(TemplateFailure) -> TemplateValue {
    switch (self, other) {
    case (.string(let lhs), .integer(let rhs)):
      .string(String(repeating: lhs, count: Int(clamping: max(rhs, 0))))
    case (.integer(let lhs), .integer(let rhs)):
      .integer(lhs &* rhs)
    default:
      .number(try self.doubleValue() * other.doubleValue())
    }
  }

  func dividing(_ other: TemplateValue) throws(TemplateFailure) -> TemplateValue {
    if case .integer(let lhs) = self, case .integer(let rhs) = other {
      return .integer(try templateIntegerDivision(lhs, rhs))
    }
    return .number(try self.doubleValue() / other.doubleValue())
  }

  func negated() throws(TemplateFailure) -> TemplateValue {
    if case .integer(let value) = self {
      return .integer(0 &- value)
    }
    return .number(-(try self.doubleValue()))
  }
}

func templateIntegerDivision(_ lhs: Int64, _ rhs: Int64) throws(TemplateFailure) -> Int64 {
  guard rhs != 0 else {
    throw TemplateFailure("division by zero")
  }
  return lhs.dividedReportingOverflow(by: rhs).partialValue
}

func templateRemainder(_ lhs: Int64, _ rhs: Int64) throws(TemplateFailure) -> Int64 {
  guard rhs != 0 else {
    throw TemplateFailure("modulo by zero")
  }
  return lhs.remainderReportingOverflow(dividingBy: rhs).partialValue
}
