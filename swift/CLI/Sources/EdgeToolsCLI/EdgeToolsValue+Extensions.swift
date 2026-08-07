import EdgeTools

extension EdgeToolsValue {
  var prettyJSONText: String {
    (try? self.encodedJSON()) ?? "<unencodable>"
  }
}
