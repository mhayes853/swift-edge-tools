import EdgeToolsCore

#if !$Embedded
  import Observation
#endif

// MARK: - EdgeToolsEngineIdentity

public final class EdgeToolsEngineIdentity: Sendable {
  public init() {}
}

// MARK: - EdgeToolsForkableModelState

public protocol EdgeToolsForkableModelState {
  func forkedContextState(copyingCache: Bool) -> sending Self
}

// MARK: - EdgeToolsTranscriptContextParameters

public struct EdgeToolsTranscriptContextParameters: Hashable, Sendable {
  public var transcript: EdgeToolsTranscript
  public var reasoningEffort: EdgeToolsReasoningEffort

  public init(
    transcript: EdgeToolsTranscript = EdgeToolsTranscript(),
    reasoningEffort: EdgeToolsReasoningEffort = .default
  ) {
    self.transcript = transcript
    self.reasoningEffort = reasoningEffort
  }
}

// MARK: - EdgeToolsTranscriptContext

public final class EdgeToolsTranscriptContext<ModelState: EdgeToolsForkableModelState>:
  Identifiable, Sendable
{
  private struct State {
    var transcript: EdgeToolsTranscript
    var reasoningEffort: EdgeToolsReasoningEffort
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

  public struct Snapshot {
    public let transcript: EdgeToolsTranscript
    public let revision: Int
    public let model: ModelState
  }

  private struct ForkSnapshot {
    let parameters: EdgeToolsTranscriptContextParameters
    let model: ModelState
  }

  private let state: Lock<State>
  public let engineIdentity: EdgeToolsEngineIdentity
  private let observationRegistrar = _ObservationRegistrar()

  public var transcript: EdgeToolsTranscript {
    get {
      self.access(.transcript)
      return self.state.withBorrowedLock { $0.transcript }
    }
    set {
      self.state.withLock { state in
        self.withMutation(of: .transcript) {
          state.transcript = newValue
          state.revision += 1
        }
      }
    }
  }

  public var reasoningEffort: EdgeToolsReasoningEffort {
    get {
      self.access(.reasoningEffort)
      return self.state.withBorrowedLock { $0.reasoningEffort }
    }
    set {
      self.state.withLock { state in
        self.withMutation(of: .reasoningEffort) {
          state.reasoningEffort = newValue
          state.revision += 1
        }
      }
    }
  }

  public var isResponding: Bool {
    self.access(.isResponding)
    return self.state.withBorrowedLock { $0.isResponding }
  }

  public init(
    parameters: EdgeToolsTranscriptContextParameters,
    model: sending ModelState,
    engineIdentity: EdgeToolsEngineIdentity
  ) {
    self.state = Lock(
      State(
        transcript: parameters.transcript,
        reasoningEffort: parameters.reasoningEffort,
        model: model
      )
    )
    self.engineIdentity = engineIdentity
  }

  public func fork() -> EdgeToolsTranscriptContext<ModelState> {
    let snapshot = self.state.withBorrowedLock { state in
      ForkSnapshot(
        parameters: EdgeToolsTranscriptContextParameters(
          transcript: state.transcript,
          reasoningEffort: state.reasoningEffort
        ),
        model: state.model.forkedContextState(copyingCache: state.isResponding)
      )
    }
    return EdgeToolsTranscriptContext(
      parameters: snapshot.parameters,
      model: snapshot.model,
      engineIdentity: self.engineIdentity
    )
  }

  public func transcript(
    appending message: EdgeToolsTranscript.UserMessage
  ) -> EdgeToolsTranscript {
    self.state.withBorrowedLock { state in
      var transcript = state.transcript
      transcript.messages.append(.user(message))
      transcript.reasoningEffort = state.reasoningEffort
      return transcript
    }
  }

  public func begin(appending message: EdgeToolsTranscript.UserMessage) throws -> Snapshot {
    try self.state.withLock { state in
      guard !state.isResponding else {
        throw EdgeToolsError.contextInUse
      }
      let model = state.model.forkedContextState(copyingCache: false)
      var transcript = state.transcript
      transcript.messages.append(.user(message))
      var snapshot: Snapshot!
      self.withMutation(of: .transcript) {
        self.withMutation(of: .isResponding) {
          state.transcript = transcript
          state.revision += 1
          state.isResponding = true
          transcript.reasoningEffort = state.reasoningEffort
          snapshot = Snapshot(
            transcript: transcript,
            revision: state.revision,
            model: model
          )
        }
      }
      return snapshot
    }
  }

  public func begin() throws -> Snapshot {
    try self.state.withLock { state in
      guard !state.isResponding else {
        throw EdgeToolsError.contextInUse
      }
      let model = state.model.forkedContextState(copyingCache: false)
      var snapshot: Snapshot!
      self.withMutation(of: .isResponding) {
        state.isResponding = true
        var transcript = state.transcript
        transcript.reasoningEffort = state.reasoningEffort
        snapshot = Snapshot(
          transcript: transcript,
          revision: state.revision,
          model: model
        )
      }
      return snapshot
    }
  }

  public func finish(
    generation: EdgeToolsEngineGeneration?,
    revision: Int,
    model: sending ModelState
  ) {
    // The compiler cannot track a `sending` parameter through the lock closure. This method
    // stores the generation branch once when its context revision still matches.
    nonisolated(unsafe) let model = model
    self.state.withLock { state in
      state.finish(
        generation: generation,
        revision: revision,
        model: model
      )
    }
    self.withMutation(of: .transcript) {
    }
    self.withMutation(of: .isResponding) {
    }
  }
}

#if !$Embedded
  extension EdgeToolsTranscriptContext: Observable {}
#endif

// MARK: - Observation

extension EdgeToolsTranscriptContext {
  fileprivate enum ObservedProperty {
    case transcript
    case reasoningEffort
    case isResponding
  }

  fileprivate func access(_ property: ObservedProperty) {
    #if !$Embedded
      switch property {
      case .transcript: self.observationRegistrar.access(self, keyPath: \.transcript)
      case .reasoningEffort: self.observationRegistrar.access(self, keyPath: \.reasoningEffort)
      case .isResponding: self.observationRegistrar.access(self, keyPath: \.isResponding)
      }
    #endif
  }

  fileprivate func withMutation<Result>(
    of property: ObservedProperty,
    _ body: () -> Result
  ) -> Result {
    #if !$Embedded
      switch property {
      case .transcript:
        self.observationRegistrar.withMutation(of: self, keyPath: \.transcript, body)
      case .reasoningEffort:
        self.observationRegistrar.withMutation(of: self, keyPath: \.reasoningEffort, body)
      case .isResponding:
        self.observationRegistrar.withMutation(of: self, keyPath: \.isResponding, body)
      }
    #else
      body()
    #endif
  }
}
