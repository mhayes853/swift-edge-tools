import EdgeToolsCore

extension EdgeToolsValue {
  init(json string: String) throws {
    try self.init(json: Array(string.utf8))
  }
}
