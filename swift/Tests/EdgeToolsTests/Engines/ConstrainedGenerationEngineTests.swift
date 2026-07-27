import CustomDump
import EdgeTools
import Testing

#if MLX && XGrammar && canImport(MLX) && !os(WASI)
  @Suite(.serialized, .enabledIfXcode())
  struct MLXConstrainedGenerationEngineTests {
    private typealias Engine = EdgeToolsMLXEngine<NeedleMLXModel>

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
      let engine = try await makeNeedleONNXEngine()
      let parameters = NeedleONNXEngine.GenerateParameters(
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
      let engine = try await makeNeedleCoreMLEngine()
      let parameters = NeedleCoreMLEngine.GenerateParameters(
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
      let engine = try await makeNeedleCoreAIEngine()
      let parameters = NeedleCoreAIEngine.GenerateParameters(
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
