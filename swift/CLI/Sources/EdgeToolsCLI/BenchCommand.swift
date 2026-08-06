import ArgumentParser
import EdgeTools
import Foundation

// MARK: - BenchCommand

struct BenchCommand: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "bench",
    abstract: "Repeatedly generate and report the distribution of performance metrics."
  )

  @OptionGroup var model: ModelOptions
  @OptionGroup var generation: GenerationOptions

  @Option(help: "How many measured runs to perform.")
  var repeatCount: Int = 10

  @Option(help: "How many unmeasured runs to perform first.")
  var warmup: Int = 2

  @Flag(help: "Emit a single JSON object instead of human-readable output.")
  var json = false

  func validate() throws {
    guard self.repeatCount > 0 else {
      throw ValidationError("--repeat-count must be at least 1.")
    }
    guard self.warmup >= 0 else {
      throw ValidationError("--warmup cannot be negative.")
    }
  }

  func run() async throws {
    let prompt = try self.generation.resolvedPrompt()
    let tools = try self.generation.toolDefinitions()
    let loaded = try await loadModel(model: self.model, quiet: self.json)
    let request = try makeRequest(
      options: self.generation,
      prompt: prompt,
      tools: tools,
      loaded: loaded,
      quiet: self.json
    )

    for _ in 0..<self.warmup {
      _ = try await loaded.runner.generate(request, channel: EdgeToolsGenerationChannel())
    }

    let clock = ContinuousClock()
    let showsProgress = !self.json && isatty(STDERR_FILENO) == 1
    var samples = [BenchSample]()
    for index in 0..<self.repeatCount {
      if showsProgress {
        FileHandle.standardError.write(Data("run \(index + 1)/\(self.repeatCount)\r".utf8))
      }
      await loaded.runner.reset()
      let start = clock.now
      let generation = try await loaded.runner.generate(
        request,
        channel: EdgeToolsGenerationChannel()
      )
      samples.append(
        BenchSample(
          endToEnd: start.duration(to: clock.now),
          generation: generation
        )
      )
    }
    if showsProgress { FileHandle.standardError.write(Data("\u{1B}[2K".utf8)) }

    let summary = BenchSummary(
      samples: samples,
      peakResidentBytes: peakResidentBytes(),
      peakGPUBytes: peakGPUBytes()
    )
    if self.json {
      output(try summary.jsonReport(loaded: loaded))
    } else {
      summary.printReport(loaded: loaded, warmup: self.warmup)
    }
  }
}

// MARK: - BenchSample

struct BenchSample {
  let endToEnd: Duration
  let generation: EdgeToolsEngineGeneration
}

// MARK: - BenchSummary

struct BenchSummary {
  let samples: [BenchSample]
  let peakResidentBytes: Int
  let peakGPUBytes: Int

  var prefillRates: [Double] { self.samples.map(\.generation.prefillMetrics.tokensPerSecond) }
  var decodeRates: [Double] { self.samples.map(\.generation.decodeMetrics.tokensPerSecond) }
  var timesToFirstToken: [Double] {
    self.samples.map(\.generation.decodeMetrics.durationToFirstToken.milliseconds)
  }
  var endToEndTimes: [Double] { self.samples.map(\.endToEnd.milliseconds) }
  var toolCallCount: Int { self.samples.count { !$0.generation.toolCalls.isEmpty } }
}

extension BenchSummary {
  func printReport(loaded: LoadedModel, warmup: Int) {
    output(
      """
      \(loaded.detection.model.displayName) · \(loaded.engine.rawValue) · \
      \(self.samples.count) runs (\(warmup) warmup)
      """
    )
    output()
    output("                   median       p95       min       max")
    printRow("Prefill tok/s", self.prefillRates, format: "%.1f")
    printRow("Decode tok/s", self.decodeRates, format: "%.1f")
    printRow("TTFT ms", self.timesToFirstToken, format: "%.0f")
    printRow("E2E ms", self.endToEndTimes, format: "%.0f")
    output()
    output("Peak RSS          \(formattedBytes(self.peakResidentBytes))")
    if self.peakGPUBytes > 0 {
      output("Peak GPU          \(formattedBytes(self.peakGPUBytes))")
    }
    output("Tool calls        \(self.toolCallCount)/\(self.samples.count) runs produced calls")
  }

  func jsonReport(loaded: LoadedModel) throws -> String {
    let report = BenchReport(
      model: loaded.detection.model.displayName,
      engine: loaded.engine.rawValue,
      runs: self.samples.count,
      prefillTokensPerSecond: .init(self.prefillRates),
      decodeTokensPerSecond: .init(self.decodeRates),
      timeToFirstTokenMilliseconds: .init(self.timesToFirstToken),
      endToEndMilliseconds: .init(self.endToEndTimes),
      peakResidentBytes: self.peakResidentBytes,
      peakGPUBytes: self.peakGPUBytes,
      runsWithToolCalls: self.toolCallCount
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
    return String(decoding: try encoder.encode(report), as: UTF8.self)
  }
}

// MARK: - BenchReport

struct BenchReport: Encodable {
  struct Distribution: Encodable {
    let median: Double
    let p95: Double
    let min: Double
    let max: Double

    init(_ values: [Double]) {
      self.median = percentile(values, 0.5)
      self.p95 = percentile(values, 0.95)
      self.min = values.min() ?? 0
      self.max = values.max() ?? 0
    }
  }

  let model: String
  let engine: String
  let runs: Int
  let prefillTokensPerSecond: Distribution
  let decodeTokensPerSecond: Distribution
  let timeToFirstTokenMilliseconds: Distribution
  let endToEndMilliseconds: Distribution
  let peakResidentBytes: Int
  let peakGPUBytes: Int
  let runsWithToolCalls: Int
}

// MARK: - Statistics

func percentile(_ values: [Double], _ fraction: Double) -> Double {
  guard !values.isEmpty else { return 0 }
  let sorted = values.sorted()
  let index = Int((Double(sorted.count - 1) * fraction).rounded())
  return sorted[index]
}

private func printRow(_ label: String, _ values: [Double], format: String) {
  let columns = [
    percentile(values, 0.5), percentile(values, 0.95), values.min() ?? 0, values.max() ?? 0
  ]
  let cells = columns.map {
    String(format: format, $0).leftPadded(to: 10)
  }
  output(label.rightPadded(to: 17) + cells.joined())
}

extension String {
  fileprivate func leftPadded(to length: Int) -> String {
    String(repeating: " ", count: Swift.max(0, length - self.count)) + self
  }

  fileprivate func rightPadded(to length: Int) -> String {
    self + String(repeating: " ", count: Swift.max(0, length - self.count))
  }
}
