import EdgeTools
import Foundation

extension Duration {
  var milliseconds: Double {
    Double(self.components.seconds) * 1000 + Double(self.components.attoseconds) / 1e15
  }

  var displayText: String {
    self.milliseconds >= 1000
      ? String(format: "%.2fs", self.milliseconds / 1000)
      : String(format: "%.0fms", self.milliseconds)
  }
}

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

extension EdgeToolsValue {
  var prettyJSONText: String {
    (try? self.encodedJSON()) ?? "<unencodable>"
  }
}

extension String {
  func leftPadded(to length: Int) -> String {
    String(repeating: " ", count: Swift.max(0, length - self.count)) + self
  }

  func rightPadded(to length: Int) -> String {
    self + String(repeating: " ", count: Swift.max(0, length - self.count))
  }
}

struct DynamicCodingKey: CodingKey {
  var stringValue: String
  var intValue: Int?

  init(_ stringValue: String) {
    self.stringValue = stringValue
  }

  init?(stringValue: String) {
    self.init(stringValue)
  }

  init?(intValue: Int) {
    self.stringValue = String(intValue)
    self.intValue = intValue
  }
}
