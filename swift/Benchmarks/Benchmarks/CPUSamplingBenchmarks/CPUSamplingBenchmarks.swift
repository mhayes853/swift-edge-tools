import Benchmark
import EdgeTools

#if canImport(Darwin)
  import Darwin
#elseif canImport(Glibc)
  import Glibc
#endif

// MARK: - Benchmarks

nonisolated(unsafe) let benchmarks = {
  Benchmark.defaultConfiguration = Benchmark.Configuration(
    metrics: [.wallClock, .instructions, .mallocCountTotal, .peakMemoryResident],
    warmupIterations: 25,
    maxDuration: .seconds(2)
  )

  for constraint in Constraint.allCases {
    for configuration in samplingConfigurations {
      registerSamplingBenchmark(
        vocabulary: .reference,
        constraint: constraint,
        configuration: configuration
      )
    }
  }

  for vocabulary in Vocabulary.allCases where vocabulary != .reference {
    for configuration in samplingConfigurations {
      registerSamplingBenchmark(
        vocabulary: vocabulary,
        constraint: .unconstrained,
        configuration: configuration
      )
    }
  }

  for vocabulary in Vocabulary.allCases {
    for configuration in samplingConfigurations {
      registerSamplingBenchmark(
        vocabulary: vocabulary,
        distribution: .flat,
        constraint: .unconstrained,
        configuration: configuration
      )
    }
  }

  for vocabulary in Vocabulary.allCases {
    for constraint in Constraint.allCases where constraint != .unconstrained {
      registerBitmaskBenchmark(vocabulary: vocabulary, constraint: constraint)
    }
  }

  for permitted in maskDensities {
    registerDensityBenchmark(vocabulary: .reference, permitted: permitted)
  }
}

private let maskDensities = [16, 64, 256, 1_024, 4_096, 16_384]

// MARK: - Vocabulary

private enum Vocabulary: String, CaseIterable {
  case tiny = "8k"
  case small = "64k"
  case medium = "128k"
  case large = "256k"

  static let reference = Self.medium

  var size: Int {
    switch self {
    case .tiny: 8_192
    case .small: 65_536
    case .medium: 131_072
    case .large: 262_144
    }
  }
}

// MARK: - Distribution

private enum Distribution: String, CaseIterable {
  case peaked
  case flat

  var bulkCentre: Float {
    switch self {
    case .peaked: -18
    case .flat: -6
    }
  }

  var headCount: Int {
    switch self {
    case .peaked: 128
    case .flat: 4096
    }
  }

  var headDecay: Float {
    switch self {
    case .peaked: 0.175
    case .flat: 0.002
    }
  }
}

// MARK: - Constraint

private enum Constraint: String, CaseIterable {
  case unconstrained
  case permissive
  case restrictive

  var permittedCount: Int? {
    switch self {
    case .unconstrained, .permissive: nil
    case .restrictive: 512
    }
  }
}

// MARK: - SamplingConfiguration

private struct SamplingConfiguration {
  let name: String
  let parameters: EdgeToolsFusedSamplingParameters
}

private let samplingConfigurations = [
  SamplingConfiguration(
    name: "greedy",
    parameters: .greedy
  ),
  SamplingConfiguration(
    name: "temperature",
    parameters: EdgeToolsFusedSamplingParameters(temperature: 0.7, seed: 8_675_309)
  ),
  SamplingConfiguration(
    name: "top-k",
    parameters: EdgeToolsFusedSamplingParameters(temperature: 0.7, topK: 20, seed: 8_675_309)
  ),
  SamplingConfiguration(
    name: "top-p",
    parameters: EdgeToolsFusedSamplingParameters(temperature: 0.7, topP: 0.95, seed: 8_675_309)
  ),
  SamplingConfiguration(
    name: "top-k+top-p+min-p",
    parameters: EdgeToolsFusedSamplingParameters(
      temperature: 0.7,
      topK: 20,
      topP: 0.95,
      minP: 0.05,
      seed: 8_675_309
    )
  ),
  SamplingConfiguration(
    name: "penalties",
    parameters: EdgeToolsFusedSamplingParameters(
      temperature: 0.7,
      topK: 20,
      repetitionPenalty: 1.1,
      presencePenalty: 0.1,
      repetitionContextSize: 64,
      seed: 8_675_309
    )
  )
]

// MARK: - Registration

private func registerSamplingBenchmark(
  vocabulary: Vocabulary,
  distribution: Distribution = .peaked,
  constraint: Constraint,
  configuration: SamplingConfiguration
) {
  let fixture = logitsFixture(for: vocabulary, distribution: distribution)
  let bitmask = grammarBitmask(vocabulary: vocabulary, constraint: constraint)
  let name = """
    sample \(vocabulary.rawValue) \(distribution.rawValue) \
    \(constraint.rawValue) \(configuration.name)
    """
  Benchmark(name) { benchmark in
    let sampler = preparedSampler(parameters: configuration.parameters, vocabulary: vocabulary)
    for _ in benchmark.scaledIterations {
      fixture.scratch.update(fromContentsOf: fixture.logits)
      benchmark.startMeasurement()
      blackHole(sampler.sample(logits: fixture.scratch, bitmask: bitmask))
      benchmark.stopMeasurement()
    }
  }
}

private func registerDensityBenchmark(vocabulary: Vocabulary, permitted: Int) {
  let fixture = logitsFixture(for: vocabulary, distribution: .peaked)
  let bitmask = permittingBitmask(vocabulary: vocabulary, permitted: permitted)
  Benchmark("density \(vocabulary.rawValue) permitted-\(permitted) greedy") { benchmark in
    let sampler = EdgeToolsCPUFusedSampler(parameters: .greedy)
    for _ in benchmark.scaledIterations {
      fixture.scratch.update(fromContentsOf: fixture.logits)
      benchmark.startMeasurement()
      blackHole(sampler.sample(logits: fixture.scratch, bitmask: bitmask))
      benchmark.stopMeasurement()
    }
  }
}

private func registerBitmaskBenchmark(vocabulary: Vocabulary, constraint: Constraint) {
  let fixture = logitsFixture(for: vocabulary, distribution: .peaked)
  guard let bitmask = grammarBitmask(vocabulary: vocabulary, constraint: constraint) else {
    return
  }
  Benchmark("bitmask \(vocabulary.rawValue) \(constraint.rawValue)") { benchmark in
    for _ in benchmark.scaledIterations {
      fixture.scratch.update(fromContentsOf: fixture.logits)
      benchmark.startMeasurement()
      applyBitmaskCPU(logits: fixture.scratch, mask: bitmask)
      benchmark.stopMeasurement()
      blackHole(fixture.scratch[0])
    }
  }
}

// MARK: - Fixtures

private struct LogitsFixture {
  let logits: [Float]
  let scratch: UnsafeMutableBufferPointer<Float>
}

private struct FixtureKey: Hashable {
  let vocabulary: Vocabulary
  let distribution: Distribution
}

private nonisolated(unsafe) var logitsFixtures = [FixtureKey: LogitsFixture]()

private func logitsFixture(
  for vocabulary: Vocabulary,
  distribution: Distribution
) -> LogitsFixture {
  let key = FixtureKey(vocabulary: vocabulary, distribution: distribution)
  if let fixture = logitsFixtures[key] {
    return fixture
  }
  let logits = syntheticLogits(count: vocabulary.size, distribution: distribution)
  let fixture = LogitsFixture(
    logits: logits,
    scratch: UnsafeMutableBufferPointer<Float>.allocate(capacity: logits.count)
  )
  logitsFixtures[key] = fixture
  return fixture
}

private func syntheticLogits(count: Int, distribution: Distribution) -> [Float] {
  var random = SplitMix64(seed: 0x5EED_1234_ABCD_9876)
  var logits = (0..<count)
    .map { _ in random.nextGaussian() * 2.5 + distribution.bulkCentre }
  var logit = Float(4)
  for _ in 0..<Swift.min(distribution.headCount, count) {
    logits[Int(random.next() % UInt64(count))] = logit
    logit -= distribution.headDecay + random.nextUniform() * distribution.headDecay
  }
  return logits
}

private func grammarBitmask(vocabulary: Vocabulary, constraint: Constraint) -> GrammarBitmask? {
  let bitCount = GrammarBitmask.bitCount(forVocabularySize: vocabulary.size)
  switch constraint {
  case .unconstrained:
    return nil
  case .permissive:
    return GrammarBitmask(bitCount: bitCount, repeating: true)
  case .restrictive:
    return permittingBitmask(vocabulary: vocabulary, permitted: constraint.permittedCount ?? 0)
  }
}

private func permittingBitmask(vocabulary: Vocabulary, permitted: Int) -> GrammarBitmask {
  var mask = GrammarBitmask(bitCount: GrammarBitmask.bitCount(forVocabularySize: vocabulary.size))
  var random = SplitMix64(seed: 0xBEEF_0F1E_2D3C_4B5A)
  for _ in 0..<permitted {
    mask[Int(random.next() % UInt64(vocabulary.size))] = true
  }
  return mask
}

private func preparedSampler(
  parameters: EdgeToolsFusedSamplingParameters,
  vocabulary: Vocabulary
) -> EdgeToolsCPUFusedSampler {
  let sampler = EdgeToolsCPUFusedSampler(parameters: parameters)
  var random = SplitMix64(seed: 0x1234_5678_9ABC_DEF0)
  sampler.history.seed(
    (0..<sampler.history.capacity).map { _ in Int(random.next() % UInt64(vocabulary.size)) }
  )
  return sampler
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
    return sqrtf(-2 * logf(uniform)) * cosf(angle)
  }
}
