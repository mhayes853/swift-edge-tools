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
