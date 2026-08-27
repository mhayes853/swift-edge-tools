import EdgeToolsCore

#if !$Embedded
  import Observation
#endif

// MARK: - EdgeToolsEngineIdentity

final class EdgeToolsEngineIdentity: Sendable {}

// MARK: - TranscriptContextStorage

final class TranscriptContextStorage<ModelState: Sendable>: Sendable {
  private struct State {
    var transcript: EdgeToolsTranscript
    var isResponding = false
    var revision = 0
    var model: ModelState

    mutating func finish(
      generation: EdgeToolsEngineGeneration?,
      revision: Int,
      model: sending ModelState
    ) {
      if self.revision == revision {
        self.model = model
      }
      if let generation {
        self.transcript.messages.append(EdgeToolsTranscript.Message(generation: generation))
        self.revision += 1
      }
      self.isResponding = false
    }
  }

  struct Snapshot: Sendable {
    let transcript: EdgeToolsTranscript
    let revision: Int
    let model: ModelState
  }

  private struct ForkSnapshot {
    let transcript: EdgeToolsTranscript
    let model: ModelState
  }

  private let state: Lock<State>
  let tools: [any EdgeTool]
  let engineIdentity: EdgeToolsEngineIdentity
  private let observationRegistrar = _ObservationRegistrar()

  var transcript: EdgeToolsTranscript {
    get {
      self.access(\.transcript)
      return self.state.withBorrowedLock { $0.transcript }
    }
    set {
      self.state.withLock { state in
        state.transcript = newValue
        state.revision += 1
      }
      self.notifyMutation(of: \.transcript)
    }
  }

  var reasoningEffort: EdgeToolsReasoningEffort {
    get {
      self.access(\.transcript)
      return self.state.withBorrowedLock { $0.transcript.reasoningEffort }
    }
    set {
      self.state.withLock { state in
        state.transcript.reasoningEffort = newValue
        state.revision += 1
      }
      self.notifyMutation(of: \.transcript)
    }
  }

  var isResponding: Bool {
    self.access(\.isResponding)
    return self.state.withBorrowedLock { $0.isResponding }
  }

  init(
    transcript: EdgeToolsTranscript,
    tools: [any EdgeTool],
    model: sending ModelState,
    engineIdentity: EdgeToolsEngineIdentity
  ) {
    self.state = Lock(
      State(
        transcript: transcript,
        model: model
      )
    )
    self.tools = tools
    self.engineIdentity = engineIdentity
  }

  func fork(
    model forkModel: @Sendable (borrowing ModelState) -> sending ModelState
  ) -> TranscriptContextStorage<ModelState> {
    let snapshot = self.state.withBorrowedLock { state in
      ForkSnapshot(
        transcript: state.transcript,
        model: forkModel(state.model)
      )
    }
    return TranscriptContextStorage(
      transcript: snapshot.transcript,
      tools: self.tools,
      model: snapshot.model,
      engineIdentity: self.engineIdentity
    )
  }

  func transcript(
    appending prompt: EdgeToolsTranscript.Prompt
  ) -> EdgeToolsTranscript {
    self.state.withBorrowedLock { state in
      var transcript = state.transcript
      transcript.messages.append(contentsOf: prompt.messages)
      return transcript
    }
  }

  func begin(appending prompt: EdgeToolsTranscript.Prompt? = nil) throws -> Snapshot {
    let snapshot = try self.state.withLock { state in
      guard !state.isResponding else {
        throw EdgeToolsError.contextInUse
      }
      if let prompt {
        state.transcript.messages.append(contentsOf: prompt.messages)
        state.revision += 1
      }
      state.isResponding = true
      return Snapshot(
        transcript: state.transcript,
        revision: state.revision,
        model: state.model
      )
    }
    if prompt != nil {
      self.notifyMutation(of: \.transcript)
    }
    self.notifyMutation(of: \.isResponding)
    return snapshot
  }

  func finish(
    generation: EdgeToolsEngineGeneration?,
    revision: Int,
    model: sending ModelState
  ) {
    let transcriptChanged = generation != nil
    self.state.withLock { $0.finish(generation: generation, revision: revision, model: model) }
    if transcriptChanged {
      self.notifyMutation(of: \.transcript)
    }
    self.notifyMutation(of: \.isResponding)
  }
}

#if !$Embedded
  extension TranscriptContextStorage: Observable {}
#endif

// MARK: - Observation

extension TranscriptContextStorage {
  fileprivate func access<Member>(
    _ keyPath: KeyPath<TranscriptContextStorage<ModelState>, Member>
  ) {
    #if !$Embedded
      self.observationRegistrar.access(self, keyPath: keyPath)
    #endif
  }

  fileprivate func notifyMutation<Member>(
    of keyPath: KeyPath<TranscriptContextStorage<ModelState>, Member>
  ) {
    #if !$Embedded
      self.observationRegistrar.willSet(self, keyPath: keyPath)
      self.observationRegistrar.didSet(self, keyPath: keyPath)
    #endif
  }
}
