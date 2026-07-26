#if ONNX && canImport(COnnxRuntime) && !os(WASI)
  import CustomDump
  import EdgeTools
  import Foundation
  import SnapshotTesting
  import Testing

  @Suite(.serialized)
  struct `NeedleONNXEngine tests` {
    @Test
    func `Generate Basics With CPU Execution Provider`() async throws {
      let engine = try await makeNeedleONNXEngine()
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let generation = try await generationTask.value

      expectNoDifference(generation.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.metadata, as: .dump, record: .all)
      }
    }

    @Test
    func `Sequential Generations With CPU`() async throws {
      let engine = try await makeNeedleONNXEngine()
      let firstTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let first = try await firstTask.value
      let secondTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let second = try await secondTask.value

      withKnownIssue {
        assertSnapshot(of: first.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
        assertSnapshot(of: second.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    func `Concurrent Generations With CPU`() async throws {
      let engine = try await makeNeedleONNXEngine()
      let firstTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let secondTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )

      let (first, second) = try await (firstTask.value, secondTask.value)
      withKnownIssue {
        assertSnapshot(of: first.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
        assertSnapshot(of: second.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    func `Generate Basics With Core ML CPU And GPU`() async throws {
      let engine = try await makeNeedleONNXEngine(
        runtimeConfiguration: .init(
          executionProviders: [.coreML(computeUnits: .cpuAndGPU)]
        )
      )
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let generation = try await generationTask.value

      expectNoDifference(generation.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.metadata, as: .dump, record: .all)
      }
    }

    @Test
    func `Sequential Generations With Core ML`() async throws {
      let engine = try await makeNeedleONNXEngine(
        runtimeConfiguration: .init(
          executionProviders: [.coreML(computeUnits: .cpuAndGPU)]
        )
      )
      let firstTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let first = try await firstTask.value
      let secondTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let second = try await secondTask.value

      withKnownIssue {
        assertSnapshot(of: first.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
        assertSnapshot(of: second.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    func `Concurrent Generations With Core ML`() async throws {
      let engine = try await makeNeedleONNXEngine(
        runtimeConfiguration: .init(
          executionProviders: [.coreML(computeUnits: .cpuAndGPU)]
        )
      )
      let firstTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let secondTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )

      let (first, second) = try await (firstTask.value, secondTask.value)
      withKnownIssue {
        assertSnapshot(of: first.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
        assertSnapshot(of: second.tokens.map(\.stringValue).joined(), as: .lines, record: .all)
      }
    }

    @Test
    func `Generate Invokes Custom Logit Processor`() async throws {
      let engine = try await makeNeedleONNXEngine()
      let processor = CountingONNXLogitsProcessor()
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: NeedleONNXEngine.GenerateParameters(processor: processor),
        channel: EdgeToolsGenerationChannel()
      )
      let generation = try await generationTask.value

      expectNoDifference(processor.promptCalls, 1)
      expectNoDifference(processor.processCalls, generation.tokens.count)
      expectNoDifference(processor.didSampleCalls, generation.tokens.count)
    }

    @Test
    func `Generate Stops And Returns Stopped Generation`() async throws {
      let engine = try await makeNeedleONNXEngine()
      let generationTaskBox = Lock<NeedleONNXEngine.GenerationTask?>(nil)
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel(
          onToken: { _ in generationTaskBox.withLock { $0?.stop() } }
        )
      )
      generationTaskBox.withLock { $0 = generationTask }
      let generation = try await generationTask.value

      expectNoDifference(generation.wasStopped, true)
      expectNoDifference(generation.tokens.count, 1)
      expectNoDifference(generation.decodeMetrics.tokens, 1)
    }

    @Test
    func `Generate Cancels And Throws Cancellation Error`() async throws {
      let engine = try await makeNeedleONNXEngine()
      let task = Task {
        let generationTask = try engine.generate(
          prompt: .sendAdventureEmail,
          tools: NeedlePrompt.sendAdventureEmailDefinitions,
          parameters: .default,
          channel: EdgeToolsGenerationChannel()
        )
        _ = try await generationTask.value
      }

      task.cancel()
      await #expect(throws: CancellationError.self) {
        _ = try await task.value
      }
    }

    @Test
    func `Generate Through EdgeToolsSession`() async throws {
      let engine = try await makeNeedleONNXEngine()
      let session = EdgeToolsSession(engine: engine)
      let generation = try await session.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailTools
      )

      expectNoDifference(generation.engineGeneration.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.engineGeneration.metadata, as: .dump, record: .all)
        assertSnapshot(
          of: generation.engineGeneration.tokens.map(\.stringValue).joined(),
          as: .lines,
          record: .all
        )
      }
    }

    @Test
    func `Generate Throws When Prompt Exceeds Context Length`() async throws {
      let engine = try await makeNeedleONNXEngine()
      let prompt = NeedlePrompt(system: "", user: String(repeating: "token ", count: 2_000))

      let error = await #expect(throws: EdgeToolsError.self) {
        let generationTask = try engine.generate(
          prompt: prompt,
          parameters: .default,
          channel: EdgeToolsGenerationChannel()
        )
        _ = try await generationTask.value
      }
      expectNoDifference(error?.code, .contextLengthExceeded)
    }

    @Test
    func `Generate Basics With INT4 Export`() async throws {
      let engine = try await makeNeedleONNXEngine(
        quantization: "int4"
      )
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let generation = try await generationTask.value

      expectNoDifference(generation.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.metadata, as: .dump, record: .all)
        assertSnapshot(
          of: generation.tokens.map(\.stringValue).joined(),
          as: .lines,
          record: .all
        )
      }
    }

    @Test
    func `Generate Basics With INT8 Export`() async throws {
      let engine = try await makeNeedleONNXEngine(
        quantization: "int8"
      )
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel()
      )
      let generation = try await generationTask.value

      expectNoDifference(generation.wasStopped, false)
      withKnownIssue {
        assertSnapshot(of: generation, as: .dump, record: .all)
        assertSnapshot(of: generation.metadata, as: .dump, record: .all)
        assertSnapshot(
          of: generation.tokens.map(\.stringValue).joined(),
          as: .lines,
          record: .all
        )
      }
    }

    @Test
    func `Initialization Throws For Missing Encoder Model`() async throws {
      let source = try await exportNeedleONNX()
      let directory = try temporaryONNXBundleCopy(from: source, excluding: ["encoder.onnx"])
      defer { try? FileManager.default.removeItem(at: directory) }

      let error = await #expect(throws: EdgeToolsONNXRuntimeError.self) {
        _ = try await NeedleONNXEngine(from: directory)
      }
      #expect(error?.message.contains("encoder.onnx") == true)
    }

    @Test
    func `Initialization Throws For Missing Decoder Model`() async throws {
      let source = try await exportNeedleONNX()
      let directory = try temporaryONNXBundleCopy(from: source, excluding: ["decoder.onnx"])
      defer { try? FileManager.default.removeItem(at: directory) }

      let error = await #expect(throws: EdgeToolsONNXRuntimeError.self) {
        _ = try await NeedleONNXEngine(from: directory)
      }
      #expect(error?.message.contains("decoder.onnx") == true)
    }

    @Test
    func `Initialization Throws For Missing External Data`() async throws {
      let source = try await exportNeedleONNX()
      let directory = try temporaryONNXBundleCopy(
        from: source,
        excluding: ["encoder.onnx.data"]
      )
      defer { try? FileManager.default.removeItem(at: directory) }

      let error = await #expect(throws: EdgeToolsONNXRuntimeError.self) {
        _ = try await NeedleONNXEngine(from: directory)
      }
      #expect(error?.message.contains("encoder.onnx.data") == true)
    }

    @Test
    func `Runtime Error Preserves ONNX Status And Message`() async throws {
      let source = try await exportNeedleONNX()
      let directory = try temporaryONNXBundleCopy(
        from: source,
        excluding: ["encoder.onnx", "encoder.onnx.data"]
      )
      defer { try? FileManager.default.removeItem(at: directory) }
      try Data("not an ONNX model".utf8).write(to: directory.appending(path: "encoder.onnx"))

      let error = await #expect(throws: EdgeToolsONNXRuntimeError.self) {
        _ = try await NeedleONNXEngine(from: directory)
      }
      expectNoDifference(error?.code, .onnxRuntime)
      expectNoDifference(error?.onnxRuntimeCode != nil, true)
      expectNoDifference(error?.message.isEmpty, false)
    }

    @Test
    func `Generate Streamed Response Matches Final Response`() async throws {
      let engine = try await makeNeedleONNXEngine()
      let streamedTokens = Lock([EdgeToolsToken]())
      let generationTask = try engine.generate(
        prompt: .sendAdventureEmail,
        tools: NeedlePrompt.sendAdventureEmailDefinitions,
        parameters: .default,
        channel: EdgeToolsGenerationChannel(
          onToken: { token in streamedTokens.withLock { $0.append(token) } }
        )
      )
      let generation = try await generationTask.value

      expectNoDifference(
        streamedTokens.withLock { $0.map(\.stringValue).joined() },
        generation.response
      )
    }
  }

  private func temporaryONNXBundleCopy(
    from source: URL,
    excluding excludedNames: Set<String>
  ) throws -> URL {
    let destination = FileManager.default.temporaryDirectory
      .appending(path: UUID().uuidString, directoryHint: .isDirectory)
    try FileManager.default.createDirectory(
      at: destination,
      withIntermediateDirectories: true
    )
    for sourceURL in try FileManager.default.contentsOfDirectory(
      at: source,
      includingPropertiesForKeys: nil
    ) where !excludedNames.contains(sourceURL.lastPathComponent) {
      try FileManager.default.createSymbolicLink(
        at: destination.appending(path: sourceURL.lastPathComponent),
        withDestinationURL: sourceURL
      )
    }
    return destination
  }

  private final class CountingONNXLogitsProcessor: EdgeToolsLogitsProcessor, Sendable {
    private struct Counts: Hashable, Sendable {
      var prompt = 0
      var process = 0
      var didSample = 0
    }

    private let counts = Lock(Counts())

    var promptCalls: Int { self.counts.withLock { $0.prompt } }
    var processCalls: Int { self.counts.withLock { $0.process } }
    var didSampleCalls: Int { self.counts.withLock { $0.didSample } }

    func prompt(_ prompt: [EdgeToolsToken.ID]) {
      self.counts.withLock { $0.prompt += 1 }
    }

    func process(logits: inout [Float]) -> [Float] {
      self.counts.withLock { $0.process += 1 }
      return logits
    }

    func didSample(token: EdgeToolsToken) {
      self.counts.withLock { $0.didSample += 1 }
    }
  }
#endif
