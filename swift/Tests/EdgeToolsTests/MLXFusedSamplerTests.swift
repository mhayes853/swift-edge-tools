#if MLX && canImport(MLX)
  import CustomDump
  import EdgeTools
  import MLX
  import Testing

  @Suite(.serialized, .enabledIfXcode())
  struct `MLXFusedSampler tests` {
    @Test
    func `Greedy Sampling Picks The Highest Logit`() {
      let sampler = MLXFusedSampler(parameters: EdgeToolsFusedSamplingParameters(temperature: 0))

      expectNoDifference(sampler.pick(from: .test), 1)
    }

    @Test
    func `Repetition Penalty Demotes An Already Sampled Token`() {
      let sampler = MLXFusedSampler(
        parameters: EdgeToolsFusedSamplingParameters(temperature: 0, repetitionPenalty: 2)
      )

      expectNoDifference(sampler.pick(from: .test), 1)
      expectNoDifference(sampler.pick(from: .test), 2)
    }

    @Test
    func `Repetition Penalty Covers A History Seeded With The Prompt`() {
      let history = MLXTokenHistory(capacity: 4)
      history.seed([1])
      let sampler = MLXFusedSampler(
        parameters: EdgeToolsFusedSamplingParameters(temperature: 0, repetitionPenalty: 2),
        history: history
      )

      expectNoDifference(sampler.pick(from: .test), 2)
    }

    @Test
    func `Resetting The History Restores The Unpenalized Token`() {
      let sampler = MLXFusedSampler(
        parameters: EdgeToolsFusedSamplingParameters(temperature: 0, repetitionPenalty: 2)
      )

      expectNoDifference(sampler.pick(from: .test), 1)
      sampler.history.reset()

      expectNoDifference(sampler.pick(from: .test), 1)
    }

    @Test
    func `Top K Only Samples The Highest Logits`() {
      let sampler = MLXFusedSampler(
        parameters: EdgeToolsFusedSamplingParameters(temperature: 2, topK: 2, seed: 7)
      )

      expectNoDifference(sampler.picks(from: .test, count: 64), [1, 2])
    }

    @Test
    func `Top P Only Samples The Nucleus`() {
      let sampler = MLXFusedSampler(
        parameters: EdgeToolsFusedSamplingParameters(temperature: 2, topP: 0.6, seed: 7)
      )

      expectNoDifference(sampler.picks(from: .test, count: 64), [1])
    }

    @Test
    func `Min P Only Samples Tokens Near The Most Probable One`() {
      let sampler = MLXFusedSampler(
        parameters: EdgeToolsFusedSamplingParameters(temperature: 2, minP: 0.5, seed: 7)
      )

      expectNoDifference(sampler.picks(from: .test, count: 64), [1])
    }

    @Test
    func `Seeding Repeats The Same Tokens`() {
      func run() -> [Int32] {
        let sampler = MLXFusedSampler(
          parameters: EdgeToolsFusedSamplingParameters(temperature: 1.5, topK: 4, seed: 99)
        )
        return (0..<16).map { _ in sampler.pick(from: .test) }
      }

      expectNoDifference(run(), run())
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
#endif
