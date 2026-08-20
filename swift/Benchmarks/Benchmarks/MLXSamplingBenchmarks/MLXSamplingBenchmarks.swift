import Benchmark
#if canImport(Darwin)
  import Darwin
#endif
import EdgeTools
import MLX
import MLXLMCommon

// MARK: - Benchmarks

nonisolated(unsafe) let benchmarks = {
  Benchmark.defaultConfiguration = Benchmark.Configuration(
    metrics: [.wallClock, .peakMemoryResident],
    warmupIterations: 25,
    maxDuration: .seconds(2)
  )
  initializeMLX()

  for vocabulary in Vocabulary.allCases {
    for configuration in samplingConfigurations {
      registerFusedBenchmark(vocabulary: vocabulary, configuration: configuration)
      registerUpstreamBenchmark(vocabulary: vocabulary, configuration: configuration)
    }
  }
}

// MARK: - Vocabulary

private enum Vocabulary: String, CaseIterable {
  case tiny = "8k"
  case small = "64k"
  case medium = "128k"
  case large = "256k"

  var size: Int {
    switch self {
    case .tiny: 8_192
    case .small: 65_536
    case .medium: 131_072
    case .large: 262_144
    }
  }
}

// MARK: - SamplingConfiguration

private struct SamplingConfiguration {
  let name: String
  let edgeTools: EdgeToolsFusedSamplingParameters
  let upstream: GenerateParameters
}

private let samplingConfigurations = [
  SamplingConfiguration(
    name: "greedy",
    edgeTools: .greedy,
    upstream: GenerateParameters(temperature: 0)
  ),
  SamplingConfiguration(
    name: "temperature",
    edgeTools: EdgeToolsFusedSamplingParameters(temperature: 0.7, seed: 8_675_309),
    upstream: GenerateParameters(temperature: 0.7, seed: 8_675_309)
  ),
  SamplingConfiguration(
    name: "top-k",
    edgeTools: EdgeToolsFusedSamplingParameters(
      temperature: 0.7,
      topK: 20,
      seed: 8_675_309
    ),
    upstream: GenerateParameters(temperature: 0.7, topK: 20, seed: 8_675_309)
  ),
  SamplingConfiguration(
    name: "top-p",
    edgeTools: EdgeToolsFusedSamplingParameters(
      temperature: 0.7,
      topP: 0.95,
      seed: 8_675_309
    ),
    upstream: GenerateParameters(temperature: 0.7, topP: 0.95, seed: 8_675_309)
  ),
  SamplingConfiguration(
    name: "top-k+top-p+min-p",
    edgeTools: EdgeToolsFusedSamplingParameters(
      temperature: 0.7,
      topK: 20,
      topP: 0.95,
      minP: 0.05,
      seed: 8_675_309
    ),
    upstream: GenerateParameters(
      temperature: 0.7,
      topP: 0.95,
      topK: 20,
      minP: 0.05,
      seed: 8_675_309
    )
  ),
  SamplingConfiguration(
    name: "penalties",
    edgeTools: EdgeToolsFusedSamplingParameters(
      temperature: 0.7,
      topK: 20,
      repetitionPenalty: 1.1,
      presencePenalty: 0.1,
      repetitionContextSize: 64,
      seed: 8_675_309
    ),
    upstream: GenerateParameters(
      temperature: 0.7,
      topK: 20,
      repetitionPenalty: 1.1,
      repetitionContextSize: 64,
      presencePenalty: 0.1,
      presenceContextSize: 64,
      seed: 8_675_309
    )
  )
]

// MARK: - Registration

private func registerFusedBenchmark(
  vocabulary: Vocabulary,
  configuration: SamplingConfiguration
) {
  let name = "fused \(vocabulary.rawValue) \(configuration.name)"
  let closure: Benchmark.BenchmarkClosure = { (benchmark: Benchmark) -> Void in
    benchmarkFused(
      benchmark,
      vocabulary: vocabulary,
      parameters: configuration.edgeTools
    )
  }
  Benchmark(name, closure: closure)
}

private func registerUpstreamBenchmark(
  vocabulary: Vocabulary,
  configuration: SamplingConfiguration
) {
  let name = "upstream \(vocabulary.rawValue) \(configuration.name)"
  let closure: Benchmark.BenchmarkClosure = { (benchmark: Benchmark) -> Void in
    benchmarkUpstream(
      benchmark,
      vocabulary: vocabulary,
      parameters: configuration.upstream
    )
  }
  Benchmark(name, closure: closure)
}

private func benchmarkFused(
  _ benchmark: Benchmark,
  vocabulary: Vocabulary,
  parameters: EdgeToolsFusedSamplingParameters
) {
  let logits = logitsFixture(for: vocabulary)
  let sampler = MLXFusedSampler(parameters: parameters)
  sampler.history.seed(promptFixture(for: vocabulary))
  for _ in benchmark.scaledIterations {
    benchmark.startMeasurement()
    let token = sampler.sample(logits: logits)
    let tokenID = token.item(Int32.self)
    benchmark.stopMeasurement()
    blackHole(tokenID)
  }
}

private func benchmarkUpstream(
  _ benchmark: Benchmark,
  vocabulary: Vocabulary,
  parameters: GenerateParameters
) {
  let logits = logitsFixture(for: vocabulary)
  let prompt = MLXArray(promptFixture(for: vocabulary).map(Int32.init))
  let sampler = parameters.sampler()
  var processor = parameters.processor()
  processor?.prompt(prompt)
  for _ in benchmark.scaledIterations {
    benchmark.startMeasurement()
    let processed = processor?.process(logits: logits) ?? logits
    let token = sampler.sample(logits: processed)
    let tokenID = token.item(Int32.self)
    processor?.didSample(token: token)
    benchmark.stopMeasurement()
    blackHole(tokenID)
  }
}

// MARK: - Fixtures

private func initializeMLX() {
  let value = MLXArray(Float.zero)
  eval(value)
}

private nonisolated(unsafe) var logitsFixtures = [Vocabulary: MLXArray]()
private nonisolated(unsafe) var promptFixtures = [Vocabulary: [Int]]()

private func logitsFixture(for vocabulary: Vocabulary) -> MLXArray {
  if let logits = logitsFixtures[vocabulary] {
    return logits
  }
  var random = SplitMix64(seed: 0x5EED_1234_ABCD_9876)
  var logits = (0..<vocabulary.size).map { _ in random.nextGaussian() * 2.5 - 18 }
  var logit = Float(4)
  for _ in 0..<Swift.min(128, vocabulary.size) {
    logits[Int(random.next() % UInt64(vocabulary.size))] = logit
    logit -= 0.175 + random.nextUniform() * 0.175
  }
  let result = MLXArray(logits)[.newAxis, 0...]
  eval(result)
  logitsFixtures[vocabulary] = result
  return result
}

private func promptFixture(for vocabulary: Vocabulary) -> [Int] {
  if let prompt = promptFixtures[vocabulary] {
    return prompt
  }
  var random = SplitMix64(seed: 0x1234_5678_9ABC_DEF0)
  let prompt = (0..<64).map { _ in Int(random.next() % UInt64(vocabulary.size)) }
  promptFixtures[vocabulary] = prompt
  return prompt
}

// MARK: - SplitMix64

private struct SplitMix64 {
  private var state: UInt64

  init(seed: UInt64) {
    self.state = seed
  }

  mutating func next() -> UInt64 {
    self.state &+= 0x9E37_79B9_7F4A_7C15
    var mixed = self.state
    mixed = (mixed ^ (mixed >> 30)) &* 0xBF58_476D_1CE4_E5B9
    mixed = (mixed ^ (mixed >> 27)) &* 0x94D0_49BB_1331_11EB
    return mixed ^ (mixed >> 31)
  }

  mutating func nextUniform() -> Float {
    Float(self.next() >> 40) * (1.0 / Float(1 << 24))
  }

  mutating func nextGaussian() -> Float {
    let uniform = Swift.max(self.nextUniform(), .leastNormalMagnitude)
    let angle = 2 * Float.pi * self.nextUniform()
    return sqrt(-2 * log(uniform)) * cos(angle)
  }
}
