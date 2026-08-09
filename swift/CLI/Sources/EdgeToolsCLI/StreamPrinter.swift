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

  func part(_ part: EdgeToolsGenerationPart) {
    switch self.mode {
    case .tokens:
      if case .text(let text) = part {
        self.output(text, "")
      }
    case .events:
      switch part {
      case .text(let text):
        self.output("\(self.elapsed) text \(text.compactJSONText)", "\n")
      case .reasoning(let reasoning):
        self.output("\(self.elapsed) reasoning \(reasoning.compactJSONText)", "\n")
      case .toolCall(let call):
        self.output("\(self.elapsed) tool-call \(call.name) \(call.arguments.compactJSONText)", "\n")
      @unknown default:
        break
      }
    case .none:
      break
    }
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
