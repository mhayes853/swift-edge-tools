import Foundation

extension Double {
  var tokenRateText: String {
    self.isFinite ? String(format: "%.1f tok/s", self) : "-"
  }
}
