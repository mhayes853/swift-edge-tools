import EdgeTools
import Foundation

final class StreamPrinter: Sendable {
  private let mode: StreamOption
  private let start: ContinuousClock.Instant
  private let clock = ContinuousClock()

  init(mode: StreamOption, start: ContinuousClock.Instant) {
    self.mode = mode
    self.start = start
  }

  func token(_ token: EdgeToolsToken) {
    switch self.mode {
    case .tokens:
      output(token.stringValue, terminator: "")
    case .events:
      output("\(self.elapsed) token \(token.stringValue.compactJSONText)")
    case .none:
      break
    }
  }

  func toolCall(_ call: EdgeRawToolCall) {
    guard self.mode == .events else { return }
    output("\(self.elapsed) tool-call \(call.name) \(call.arguments.compactJSONText)")
  }

  func finish() {
    if self.mode == .tokens {
      output()
    }
  }

  private var elapsed: String {
    self.start.duration(to: self.clock.now).displayText
      .padding(toLength: 8, withPad: " ", startingAt: 0)
  }
}
