import CustomDump
import EdgeTools
import Testing

#if MLX && XGrammar && canImport(MLX) && !os(WASI)
  @Suite(.serialized, .enabledIfXcode())
  struct MLXConstrainedGenerationEngineTests {
    private typealias Engine = NeedleMLXModelEngine

    @Test
    func `Generate With Explicit Grammar Constraint`() async throws {
      let engine = try await Engine(from: downloadNeedle())
      let parameters = Engine.GenerateParameters(
        constraint: .grammar(try .literal("OK"))
      )
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: parameters,
        channel: EdgeToolsGenerationChannel()
      )
      let generation = try await generationTask.value

      expectNoDifference(generation.response, "OK")
    }
  }
#endif

#if ONNX && canImport(COnnxRuntime) && !os(WASI)
  @Suite(.serialized)
  struct ONNXConstrainedGenerationEngineTests {
    @Test
    func `Generate With Explicit Grammar Constraint`() async throws {
      let engine = try await makeNeedleONNXModelEngine()
      let parameters = EdgeToolsONNXGenerateParameters(
        constraint: .grammar(try .literal("OK"))
      )
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: parameters,
        channel: EdgeToolsGenerationChannel()
      )
      let generation = try await generationTask.value

      expectNoDifference(generation.response, "OK")
    }
  }
#endif

#if CoreML && canImport(CoreML) && !os(WASI)
  @Suite(.serialized, .experimental())
  struct CoreMLConstrainedGenerationEngineTests {
    @Test
    @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
    func `Generate With Explicit Grammar Constraint`() async throws {
      let engine = try await makeNeedleCoreMLModelEngine()
      let parameters = NeedleCoreMLModelEngine.GenerateParameters(
        constraint: .grammar(try .literal("OK"))
      )
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: parameters,
        channel: EdgeToolsGenerationChannel()
      )
      let generation = try await generationTask.value

      expectNoDifference(generation.response, "OK")
    }
  }
#endif

#if swift(>=6.4) && CoreAI && canImport(CoreAI) && !os(WASI)
  @Suite(.serialized, .experimental())
  struct CoreAIConstrainedGenerationEngineTests {
    @Test
    @available(anyAppleOS 27.0, *)
    func `Generate With Explicit Grammar Constraint`() async throws {
      let engine = try await makeNeedleCoreAIModelEngine()
      let parameters = NeedleCoreAIModelEngine.GenerateParameters(
        constraint: .grammar(try .literal("OK"))
      )
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: parameters,
        channel: EdgeToolsGenerationChannel()
      )
      let generation = try await generationTask.value

      expectNoDifference(generation.response, "OK")
    }
  }
#endif
