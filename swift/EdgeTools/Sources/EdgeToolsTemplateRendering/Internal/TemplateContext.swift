// MARK: - TemplateContext

final class TemplateContext {
  private var values = [String: TemplateValue]()
  private let parent: TemplateContext?
  private let providesBuiltins: Bool

  init(parent: TemplateContext?) {
    self.parent = parent
    self.providesBuiltins = false
  }

  private init() {
    self.parent = nil
    self.providesBuiltins = true
  }

  static func builtins() -> TemplateContext {
    TemplateContext()
  }

  func value(named name: String) -> TemplateValue? {
    if let value = self.values[name] {
      return value
    }
    if self.providesBuiltins, let function = builtinFunction(named: name) {
      let value = TemplateValue.function(function)
      self.values[name] = value
      return value
    }
    return self.parent?.value(named: name)
  }

  func value(for key: TemplateValue) -> TemplateValue {
    guard case .string(let name) = key else {
      return .null
    }
    return self.value(named: name) ?? .null
  }

  func set(_ name: String, _ value: TemplateValue) {
    self.values[name] = value
  }
}
