import Foundation

extension Encodable {
  func encodedJSON(
    formatting: JSONEncoder.OutputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
  ) throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = formatting
    return String(decoding: try encoder.encode(self), as: UTF8.self)
  }

  var compactJSONText: String {
    (try? self.encodedJSON(formatting: [.withoutEscapingSlashes])) ?? "<unencodable>"
  }
}
