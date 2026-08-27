#if MLX && canImport(MLX)
  import CustomDump
  import EdgeTools
  import MLX
  import Testing

  @Suite(.serialized, .enabledIfMLXTests())
  struct `MLXFusedSampler tests` {
    @Test(arguments: FusedSamplerBehavior.allCases)
    func `Shared Sampling Semantics`(_ behavior: FusedSamplerBehavior) {
      expectFusedSamplerBehavior(behavior, using: MLXFusedSamplerDriver.self)
    }

    @Test
    func `Reseeding History Replaces Earlier Tokens`() {
      let history = MLXTokenHistory(capacity: 4)
      history.seed([1])
      history.seed([2])
      let sampler = MLXFusedSampler(
        parameters: EdgeToolsFusedSamplingParameters(temperature: 0, repetitionPenalty: 2),
        history: history
      )

      expectNoDifference(sampler.pick(from: .test), 1)
    }

    @Test
    func `Positive Top P Retains The Highest Probability Token`() {
      let sampler = MLXFusedSampler(
        parameters: EdgeToolsFusedSamplingParameters(
          temperature: 2,
          topP: 1e-8,
          seed: 7
        )
      )

      expectNoDifference(sampler.picks(from: .test, count: 8), [1])
    }
  }

  extension MLXArray {
    fileprivate static var test: MLXArray {
      MLXArray([1 as Float, 5, 3, 2])[.newAxis, 0...]
    }
  }

  extension MLXFusedSampler {
    fileprivate func pick(from logits: MLXArray) -> Int32 {
      self.sample(logits: logits).item(Int32.self)
    }

    fileprivate func picks(from logits: MLXArray, count: Int) -> [Int32] {
      Set((0..<count).map { _ in self.pick(from: logits) }).sorted()
    }
  }

  private struct MLXFusedSamplerDriver: FusedSamplerDriver {
    let sampler: MLXFusedSampler

    init(parameters: EdgeToolsFusedSamplingParameters, seededWith tokenIds: [Int]) {
      let history = MLXTokenHistory(capacity: 20)
      history.seed(tokenIds)
      self.sampler = MLXFusedSampler(parameters: parameters, history: history)
    }

    func pick(from logits: [Float]) -> Int {
      Int(self.sampler.sample(logits: MLXArray(logits)[.newAxis, 0...]).item(Int32.self))
    }

    func resetHistory() {
      self.sampler.history.reset()
    }
  }
#endif
