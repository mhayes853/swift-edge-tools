import CustomDump
import EdgeTools
import Testing

#if canImport(Darwin)
  import Darwin
#elseif canImport(Glibc)
  import Glibc
#elseif canImport(Musl)
  import Musl
#elseif canImport(WASILibc)
  import WASILibc
#elseif canImport(Android)
  import Android
#elseif canImport(ucrt)
  import ucrt
#endif

@Suite
struct `EdgeToolsCPUFusedSampler tests` {
  @Test(arguments: FusedSamplerBehavior.allCases)
  func `Shared Sampling Semantics`(_ behavior: FusedSamplerBehavior) {
    expectFusedSamplerBehavior(behavior, using: CPUFusedSamplerDriver.self)
  }

  @Test
  func `Sampling Over A Large Vocabulary Picks The Highest Logit`() {
    let sampler = EdgeToolsCPUFusedSampler(
      parameters: EdgeToolsFusedSamplingParameters(temperature: 0)
    )
    var logits = [Float](repeating: -10, count: 100_003)
    logits[51_337] = 4

    expectNoDifference(sampler.pick(from: logits), 51_337)
  }

  @Test
  func `Bitmask Excludes The Highest Logit From Sampling`() {
    let sampler = EdgeToolsCPUFusedSampler(
      parameters: EdgeToolsFusedSamplingParameters(temperature: 0)
    )
    var bitmask = GrammarBitmask(bitCount: 8, repeating: true)
    bitmask[1] = false
    var logits = [Float].test

    expectNoDifference(sampler.sample(logits: &logits, bitmask: bitmask).tokenId, 2)
  }

  @Test
  func `Sparse Bitmask Samples A Permitted Token Over A Large Vocabulary`() {
    let sampler = EdgeToolsCPUFusedSampler(
      parameters: EdgeToolsFusedSamplingParameters(temperature: 0)
    )
    var bitmask = GrammarBitmask(bitCount: GrammarBitmask.bitCount(forVocabularySize: 100_003))
    bitmask[51_337] = true
    bitmask[90_001] = true
    var logits = [Float](repeating: -10, count: 100_003)
    logits[70_000] = 12
    logits[90_001] = 4
    logits[51_337] = 2

    expectNoDifference(sampler.sample(logits: &logits, bitmask: bitmask).tokenId, 90_001)
  }

  @Test
  func `Bitmask Permitting More Tokens Than The Gather Limit Still Excludes Masked Ones`() {
    let sampler = EdgeToolsCPUFusedSampler(
      parameters: EdgeToolsFusedSamplingParameters(temperature: 0)
    )
    var bitmask = GrammarBitmask(bitCount: GrammarBitmask.bitCount(forVocabularySize: 100_003))
    for tokenId in 0..<5_000 {
      bitmask[tokenId] = true
    }
    var logits = [Float](repeating: -10, count: 100_003)
    logits[70_000] = 12
    logits[4_999] = 3

    expectNoDifference(sampler.sample(logits: &logits, bitmask: bitmask).tokenId, 4_999)
  }

  @Test
  func `Repetition Penalty Demotes A Token Reached Through A Sparse Bitmask`() {
    let sampler = EdgeToolsCPUFusedSampler(
      parameters: EdgeToolsFusedSamplingParameters(temperature: 0, repetitionPenalty: 2)
    )
    var bitmask = GrammarBitmask(bitCount: GrammarBitmask.bitCount(forVocabularySize: 100_003))
    bitmask[51_337] = true
    bitmask[90_001] = true
    var logits = [Float](repeating: -10, count: 100_003)
    logits[90_001] = 4
    logits[51_337] = 2

    expectNoDifference(sampler.sample(logits: &logits, bitmask: bitmask).tokenId, 90_001)
    expectNoDifference(sampler.sample(logits: &logits, bitmask: bitmask).tokenId, 51_337)
  }

  @Test
  func `Confidence Reflects The Margin Between The Top Two Logits`() {
    let sampler = EdgeToolsCPUFusedSampler(
      parameters: EdgeToolsFusedSamplingParameters(temperature: 0)
    )
    var logits = [Float].test

    let sample = sampler.sample(logits: &logits)
    expectNoDifference(sample.confidence, Float(1 / (1 + exp(-2.0))))
  }

  @Test
  func `Confidence Reflects The Penalized Distribution The Sample Came From`() {
    let sampler = EdgeToolsCPUFusedSampler(
      parameters: EdgeToolsFusedSamplingParameters(temperature: 0, repetitionPenalty: 2)
    )
    var first = [Float].test
    expectNoDifference(sampler.sample(logits: &first).tokenId, 1)

    var second = [Float].test
    let sample = sampler.sample(logits: &second)
    expectNoDifference(sample.tokenId, 2)
    expectNoDifference(sample.confidence, Float(1 / (1 + exp(-0.5))))
  }

  @Test
  func `Confidence Measures The Margin To The Next Distinct Logit When The Top Logits Tie`() {
    let sampler = EdgeToolsCPUFusedSampler(
      parameters: EdgeToolsFusedSamplingParameters(temperature: 0)
    )
    var logits: [Float] = [1, 5, 5, 2]

    let sample = sampler.sample(logits: &logits)
    expectNoDifference(sample.tokenId, 1)
    expectNoDifference(sample.confidence, Float(1 / (1 + exp(-3.0))))
  }

  @Test
  func `Confidence Is Computed After Applying The Bitmask`() {
    let sampler = EdgeToolsCPUFusedSampler(
      parameters: EdgeToolsFusedSamplingParameters(temperature: 0)
    )
    var bitmask = GrammarBitmask(bitCount: 8, repeating: true)
    bitmask[1] = false
    var logits = [Float].test

    let sample = sampler.sample(logits: &logits, bitmask: bitmask)
    expectNoDifference(sample.confidence, Float(1 / (1 + exp(-1.0))))
  }
}

extension [Float] {
  fileprivate static var test: [Float] {
    [1, 5, 3, 2]
  }
}

extension EdgeToolsCPUFusedSampler {
  fileprivate func sample(
    logits: inout [Float],
    bitmask: GrammarBitmask? = nil
  ) -> EdgeToolsCPUSample {
    logits.withUnsafeMutableBufferPointer { logits in
      var span = MutableSpan<Float>(_unsafeElements: logits)
      return self.sample(logits: &span, bitmask: bitmask)
    }
  }

  fileprivate func pick(from logits: [Float]) -> Int {
    var logits = logits
    return self.sample(logits: &logits).tokenId
  }
}

private struct CPUFusedSamplerDriver: FusedSamplerDriver {
  let sampler: EdgeToolsCPUFusedSampler

  init(parameters: EdgeToolsFusedSamplingParameters, seededWith tokenIds: [Int]) {
    let history = EdgeToolsCPUTokenHistory(capacity: 20)
    history.seed(tokenIds)
    self.sampler = EdgeToolsCPUFusedSampler(parameters: parameters, history: history)
  }

  func pick(from logits: [Float]) -> Int {
    self.sampler.pick(from: logits)
  }

  func resetHistory() {
    self.sampler.history.reset()
  }
}

enum FusedSamplerBehavior: CaseIterable, Sendable {
  case greedy
  case repetitionPenalty
  case promptRepetitionPenalty
  case presencePenalty
  case historyReset
  case topK
  case topP
  case minP
  case deterministicSeed
}

protocol FusedSamplerDriver {
  init(parameters: EdgeToolsFusedSamplingParameters, seededWith tokenIds: [Int])
  func pick(from logits: [Float]) -> Int
  func resetHistory()
}

func expectFusedSamplerBehavior<Driver: FusedSamplerDriver>(
  _ behavior: FusedSamplerBehavior,
  using driver: Driver.Type
) {
  let logits: [Float] = [1, 5, 3, 2]
  switch behavior {
  case .greedy:
    let sampler = Driver(
      parameters: EdgeToolsFusedSamplingParameters(temperature: 0),
      seededWith: []
    )
    expectNoDifference(sampler.pick(from: logits), 1)
  case .repetitionPenalty:
    let sampler = Driver(
      parameters: EdgeToolsFusedSamplingParameters(temperature: 0, repetitionPenalty: 2),
      seededWith: []
    )
    expectNoDifference([sampler.pick(from: logits), sampler.pick(from: logits)], [1, 2])
  case .promptRepetitionPenalty:
    let sampler = Driver(
      parameters: EdgeToolsFusedSamplingParameters(temperature: 0, repetitionPenalty: 2),
      seededWith: [1]
    )
    expectNoDifference(sampler.pick(from: logits), 2)
  case .presencePenalty:
    let sampler = Driver(
      parameters: EdgeToolsFusedSamplingParameters(temperature: 0, presencePenalty: 3),
      seededWith: []
    )
    expectNoDifference([sampler.pick(from: logits), sampler.pick(from: logits)], [1, 2])
  case .historyReset:
    let sampler = Driver(
      parameters: EdgeToolsFusedSamplingParameters(temperature: 0, repetitionPenalty: 2),
      seededWith: []
    )
    expectNoDifference(sampler.pick(from: logits), 1)
    sampler.resetHistory()
    expectNoDifference(sampler.pick(from: logits), 1)
  case .topK:
    let sampler = Driver(
      parameters: EdgeToolsFusedSamplingParameters(temperature: 2, topK: 2, seed: 7),
      seededWith: []
    )
    expectNoDifference(picks(using: sampler, from: logits), [1, 2])
  case .topP:
    let sampler = Driver(
      parameters: EdgeToolsFusedSamplingParameters(temperature: 2, topP: 0.6, seed: 7),
      seededWith: []
    )
    expectNoDifference(picks(using: sampler, from: logits), [1])
  case .minP:
    let sampler = Driver(
      parameters: EdgeToolsFusedSamplingParameters(temperature: 2, minP: 0.5, seed: 7),
      seededWith: []
    )
    expectNoDifference(picks(using: sampler, from: logits), [1])
  case .deterministicSeed:
    func run() -> [Int] {
      let sampler = Driver(
        parameters: EdgeToolsFusedSamplingParameters(temperature: 1.5, topK: 4, seed: 99),
        seededWith: []
      )
      return (0..<16).map { _ in sampler.pick(from: logits) }
    }
    expectNoDifference(run(), run())
  }
}

private func picks<Driver: FusedSamplerDriver>(
  using sampler: Driver,
  from logits: [Float]
) -> [Int] {
  Set((0..<64).map { _ in sampler.pick(from: logits) }).sorted()
}
