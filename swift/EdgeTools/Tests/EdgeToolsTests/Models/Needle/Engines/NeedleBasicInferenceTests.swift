import CustomDump
import EdgeTools
import Testing

#if ONNX && canImport(COnnxRuntime) && !os(WASI)
  @Suite(.serialized, .basicNeedleInference())
  struct `NeedleONNXBasicInference tests` {
    @Test
    func `Generate Basics With CPU Execution Provider`() async throws {
      let engine = try await makeNeedleONNXModelEngine()
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let generation = try await generationTask.value

      expectNoDifference(generation.wasStopped, false)
      expectNoDifference(generation.tokens.isEmpty, false)
    }
  }
#endif
