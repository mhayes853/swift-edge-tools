#if XGrammar
  import EdgeToolsXGrammar
#endif

#if MLX && canImport(MLX)
  import _EdgeToolsFoundation
  import MLX
  import MLXLLM
  import MLXLMCommon
  import MLXNN
  import Observation

  #if canImport(CoreImage) && canImport(MLXVLM)
    import CoreImage
    import Foundation
    import MLXVLM
  #endif

  // MARK: - MLXContextParameters

  public struct MLXContextParameters: Hashable, Sendable {
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

  // MARK: - MLXContext

  fileprivate final class MLXEngineIdentity: Sendable {}

  public final class MLXContext<Profile: MLXModelProfile>: Identifiable, Sendable
  where Profile.Prompt == EdgeToolsTranscript {
    private struct State {
      var transcript: EdgeToolsTranscript
      var reasoningEffort: EdgeToolsReasoningEffort
      var isResponding = false
      var revision = 0
      var model: MLXModelState<Profile>

      mutating func finish(
        generation: EdgeToolsEngineGeneration?,
        revision: Int,
        model: sending MLXModelState<Profile>
      ) {
        if self.revision == revision {
          self.model = model
        }
        if let generation {
          self.transcript.messages.append(.init(generation: generation))
          self.revision += 1
        }
        self.isResponding = false
      }
    }

    struct GenerationSnapshot {
      let transcript: EdgeToolsTranscript
      let revision: Int
      let model: MLXModelState<Profile>
    }

    private struct ForkSnapshot {
      let parameters: MLXContextParameters
      let model: MLXModelState<Profile>
    }

    private let state: Lock<State>
    fileprivate let engineIdentity: MLXEngineIdentity
    private let observationRegistrar = _ObservationRegistrar()

    public var transcript: EdgeToolsTranscript {
      get {
        self.observationRegistrar.access(self, keyPath: \.transcript)
        return self.state.withBorrowedLock { $0.transcript }
      }
      set {
        self.state.withLock { state in
          self.observationRegistrar.withMutation(of: self, keyPath: \.transcript) {
            state.transcript = newValue
            state.revision += 1
          }
        }
      }
    }

    public var reasoningEffort: EdgeToolsReasoningEffort {
      get {
        self.observationRegistrar.access(self, keyPath: \.reasoningEffort)
        return self.state.withBorrowedLock { $0.reasoningEffort }
      }
      set {
        self.state.withLock { state in
          self.observationRegistrar.withMutation(of: self, keyPath: \.reasoningEffort) {
            state.reasoningEffort = newValue
            state.revision += 1
          }
        }
      }
    }

    public var isResponding: Bool {
      self.observationRegistrar.access(self, keyPath: \.isResponding)
      return self.state.withBorrowedLock { $0.isResponding }
    }

    fileprivate init(
      parameters: MLXContextParameters,
      model: sending MLXModelState<Profile>,
      engineIdentity: MLXEngineIdentity
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

    public func fork() -> MLXContext<Profile> {
      let snapshot = self.state.withBorrowedLock { state in
        ForkSnapshot(
          parameters: MLXContextParameters(
            transcript: state.transcript,
            reasoningEffort: state.reasoningEffort
          ),
          model: state.model.forkedContextState(copyingCache: state.isResponding)
        )
      }
      return MLXContext(
        parameters: snapshot.parameters,
        model: snapshot.model,
        engineIdentity: self.engineIdentity
      )
    }

    func transcript(
      appending message: EdgeToolsTranscript.UserMessage
    ) -> EdgeToolsTranscript {
      self.state.withBorrowedLock { state in
        var transcript = state.transcript
        transcript.messages.append(.user(message))
        transcript.reasoningEffort = state.reasoningEffort
        return transcript
      }
    }

    func begin(appending message: EdgeToolsTranscript.UserMessage) throws -> GenerationSnapshot {
      try self.state.withLock { state in
        guard !state.isResponding else {
          throw EdgeToolsError.contextInUse
        }
        let model = state.model.forkedContextState(copyingCache: false)
        var transcript = state.transcript
        transcript.messages.append(.user(message))
        var snapshot: GenerationSnapshot!
        self.observationRegistrar.withMutation(of: self, keyPath: \.transcript) {
          self.observationRegistrar.withMutation(of: self, keyPath: \.isResponding) {
            state.transcript = transcript
            state.revision += 1
            state.isResponding = true
            transcript.reasoningEffort = state.reasoningEffort
            snapshot = GenerationSnapshot(
              transcript: transcript,
              revision: state.revision,
              model: model
            )
          }
        }
        return snapshot
      }
    }

    func begin() throws -> GenerationSnapshot {
      try self.state.withLock { state in
        guard !state.isResponding else {
          throw EdgeToolsError.contextInUse
        }
        let model = state.model.forkedContextState(copyingCache: false)
        var snapshot: GenerationSnapshot!
        self.observationRegistrar.withMutation(of: self, keyPath: \.isResponding) {
          state.isResponding = true
          var transcript = state.transcript
          transcript.reasoningEffort = state.reasoningEffort
          snapshot = GenerationSnapshot(
            transcript: transcript,
            revision: state.revision,
            model: model
          )
        }
        return snapshot
      }
    }

    func finish(
      generation: EdgeToolsEngineGeneration?,
      revision: Int,
      model: sending MLXModelState<Profile>
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
      self.observationRegistrar.withMutation(of: self, keyPath: \.transcript) {
      }
      self.observationRegistrar.withMutation(of: self, keyPath: \.isResponding) {
      }
    }
  }

  extension MLXContext: Observable {}
#endif
