#if !os(WASI)
  import CustomDump
  import EdgeTools
  import Testing

  func startNeedleGeneration<Engine: EdgeToolsEngine>(
    using engine: Engine,
    prompt: NeedlePrompt = .sendAdventureEmail,
    tools: [EdgeToolDefinition] = NeedlePrompt.sendAdventureEmailDefinitions,
    parameters: sending Engine.GenerateParameters = .default,
    channel: sending EdgeToolsGenerationChannel = EdgeToolsGenerationChannel()
  ) throws -> Engine.GenerationTask where Engine.Prompt == NeedlePrompt {
    try engine.generate(
      prompt: prompt,
      tools: tools,
      parameters: parameters,
      channel: channel
    )
  }

  func generateNeedle<Engine: EdgeToolsEngine>(
    using engine: Engine,
    prompt: NeedlePrompt = .sendAdventureEmail,
    tools: [EdgeToolDefinition] = NeedlePrompt.sendAdventureEmailDefinitions,
    parameters: sending Engine.GenerateParameters = .default,
    channel: sending EdgeToolsGenerationChannel = EdgeToolsGenerationChannel()
  ) async throws -> EdgeToolsEngineGeneration where Engine.Prompt == NeedlePrompt {
    let task = try startNeedleGeneration(
      using: engine,
      prompt: prompt,
      tools: tools,
      parameters: parameters,
      channel: channel
    )
    return try await task.value
  }

  func generateNeedleConcurrently<Engine: EdgeToolsEngine>(
    using engine: Engine
  ) async throws -> (EdgeToolsEngineGeneration, EdgeToolsEngineGeneration)
  where Engine.Prompt == NeedlePrompt {
    let first = try startNeedleGeneration(using: engine)
    let second = try startNeedleGeneration(using: engine)
    return try await (first.value, second.value)
  }

  func generateNeedleSequentially<Engine: EdgeToolsEngine>(
    using engine: Engine
  ) async throws -> (EdgeToolsEngineGeneration, EdgeToolsEngineGeneration)
  where Engine.Prompt == NeedlePrompt {
    let first = try await generateNeedle(using: engine)
    let second = try await generateNeedle(using: engine)
    return (first, second)
  }

  func expectNeedleStreamedResponseMatchesFinalResponse<Engine: EdgeToolsEngine>(
    using engine: Engine
  ) async throws where Engine.Prompt == NeedlePrompt {
    let tokens = LockBox([EdgeToolsToken]())
    let generation = try await generateNeedle(
      using: engine,
      channel: EdgeToolsGenerationChannel(
        onToken: { token in tokens.withLock { $0.append(token) } }
      )
    )

    expectNoDifference(
      tokens.withLock { $0.map(\.stringValue).joined() },
      generation.response
    )
  }

  @discardableResult
  func expectNeedleGenerationStops<Engine: EdgeToolsEngine>(
    using engine: Engine
  ) async throws -> EdgeToolsEngineGeneration where Engine.Prompt == NeedlePrompt {
    let tokens = LockBox([EdgeToolsToken]())
    let taskBox = LockBox<Engine.GenerationTask?>(nil)
    let task = try startNeedleGeneration(
      using: engine,
      channel: EdgeToolsGenerationChannel(
        onToken: { token in
          tokens.withLock { $0.append(token) }
          taskBox.withLock { $0?.stop() }
        }
      )
    )
    taskBox.withLock { $0 = task }
    let generation = try await task.value

    expectNoDifference(generation.wasStopped, true)
    let tokenCount = tokens.withLock { $0.count }
    expectNoDifference(tokenCount > 0, true)
    expectNoDifference(generation.decodeMetrics.tokens, tokenCount)
    expectNoDifference(generation.tokens.isEmpty, false)
    return generation
  }

  func expectNeedleGenerationCancellation<Engine: EdgeToolsEngine>(
    using engine: Engine
  ) async where Engine.Prompt == NeedlePrompt {
    let task = Task {
      _ = try await generateNeedle(using: engine)
    }

    task.cancel()
    await #expect(throws: CancellationError.self) {
      _ = try await task.value
    }
  }

  func generateNeedleThroughSession<Engine: EdgeToolsEngine>(
    using engine: Engine
  ) async throws -> EdgeToolsSessionGeneration where Engine.Prompt == NeedlePrompt {
    let session = EdgeToolsSession(engine: engine, tools: NeedlePrompt.sendAdventureEmailTools)
    return try await session.generate(prompt: .sendAdventureEmail)
  }

  func expectNeedleContextLengthExceeded<Engine: EdgeToolsEngine>(
    using engine: Engine
  ) async where Engine.Prompt == NeedlePrompt {
    let prompt = NeedlePrompt(
      system: "",
      user: String(repeating: "token ", count: 2_000)
    )

    let error = await #expect(throws: EdgeToolsError.self) {
      _ = try await generateNeedle(using: engine, prompt: prompt, tools: [])
    }
    expectNoDifference(error?.code, .contextLengthExceeded)
  }
#endif
