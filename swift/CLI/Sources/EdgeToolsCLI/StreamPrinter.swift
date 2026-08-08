import EdgeTools
import Foundation

final class StreamPrinter: Sendable {
  private let mode: StreamOption
  private let start: ContinuousClock.Instant
  private let clock = ContinuousClock()
  private let output: @Sendable (String, String) -> Void

  init(
    mode: StreamOption,
    start: ContinuousClock.Instant,
    output: @escaping @Sendable (String, String) -> Void
  ) {
    self.mode = mode
    self.start = start
    self.output = output
  }

  func token(_ token: EdgeToolsToken) {
    switch self.mode {
    case .tokens:
      self.output(token.stringValue, "")
    case .events:
      self.output("\(self.elapsed) token \(token.stringValue.compactJSONText)", "\n")
    case .none:
      break
    }
  }

  func toolCall(_ call: EdgeRawToolCall) {
    guard self.mode == .events else { return }
    self.output("\(self.elapsed) tool-call \(call.name) \(call.arguments.compactJSONText)", "\n")
  }

  func finish() {
    if self.mode == .tokens {
      self.output("", "\n")
    }
  }

  private var elapsed: String {
    self.start.duration(to: self.clock.now).displayText
      .padding(toLength: 8, withPad: " ", startingAt: 0)
  }
}
