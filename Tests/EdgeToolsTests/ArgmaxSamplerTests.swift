#if swift(>=6.4) && CoreAI && canImport(CoreAI)
  import CoreAI
  import CustomDump
  @testable import EdgeTools
  import Testing

  @Suite
  struct `ArgmaxSampler tests` {
    @Test(
      arguments: [
        ([3], 0),
        ([1, 4, 2], 1),
        ([4, 4, 2], 0),
        (Array(repeating: -Float.infinity, count: 33), 0)
      ]
    )
    @available(anyAppleOS 27.0, *)
    func `Samples Expected Token`(values: [Float], expectedToken: EdgeToolsToken.ID) {
      let logits = NDArray(scalars: values, shape: [1, values.count])
      let token = CoreAIArgmaxSampler().sample(logits: logits)

      expectNoDifference(token, expectedToken)
    }

    @Test
    @available(anyAppleOS 27.0, *)
    func `Samples Tail Token`() {
      var values = Array(repeating: Float(-1), count: 67)
      values[66] = 1
      let logits = NDArray(scalars: values, shape: [1, values.count])

      let token = CoreAIArgmaxSampler().sample(logits: logits)

      expectNoDifference(token, 66)
    }

    @Test
    @available(anyAppleOS 27.0, *)
    func `Samples Token From Two Hundred Fifty Six Thousand Vocabulary`() {
      var values = Array(repeating: Float(-1), count: 256 * 1024)
      values[values.count - 17] = 1
      let logits = NDArray(scalars: values, shape: [1, values.count])

      let token = CoreAIArgmaxSampler().sample(logits: logits)

      expectNoDifference(token, values.count - 17)
    }
  }
#endif
