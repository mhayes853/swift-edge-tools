import EdgeTools
import Foundation

extension Double {
  var tokenRateText: String {
    self.isFinite ? String(format: "%.1f tok/s", self) : "-"
  }
}

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
