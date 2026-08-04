import CustomDump
import EdgeTools
import Testing

#if ONNX && canImport(COnnxRuntime) && !os(WASI)
  @Suite(.serialized, .basicNeedleInference())
  struct `Needle ONNX basic inference tests` {
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

#if CoreML && canImport(CoreML) && !os(WASI)
  import CoreML

  @Suite(.serialized, .basicNeedleInference())
  struct `Needle Core ML basic inference tests` {
    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate Basics`() async throws {
      let engine = try await makeNeedleCoreMLModelEngine(computeUnits: .cpuOnly)
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

#if swift(>=6.4) && CoreAI && canImport(CoreAI) && !os(WASI)
  import CoreAI

  @Suite(.serialized, .basicNeedleInference())
  struct `Needle Core AI basic inference tests` {
    @Test
    @available(anyAppleOS 27.0, *)
    func `Generate Basics`() async throws {
      let engine = try await makeNeedleCoreAIModelEngine()
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
