#if Llama && canImport(CLlama)
  import EdgeToolsCore
  import EdgeToolsLlama

  // MARK: - LlamaEvaluationOutput

  enum LlamaEvaluationOutput {
    case none
    case lastTokenLogits

    var wantsLogits: Bool {
      self == .lastTokenLogits
    }
  }

  // MARK: - LlamaSequenceLease

  final class LlamaSequenceLease: Sendable {
    let store: LlamaKVSequenceStore
    let sequenceId: Int

    fileprivate init(store: LlamaKVSequenceStore, sequenceId: Int) {
      self.store = store
      self.sequenceId = sequenceId
    }

    deinit { self.store.release(sequenceId: self.sequenceId) }
  }

  // MARK: - LlamaKVSequenceStore

  final class LlamaKVSequenceStore: Sendable {
    private struct CachedInput {
      var units = [LlamaPreparedInputUnit]()
      var positionCount = 0

      mutating func append(tokenIds: some Collection<EdgeToolsToken.ID>) {
        self.units.append(contentsOf: tokenIds.map { LlamaPreparedInputUnit(value: .token($0)) })
        self.positionCount += tokenIds.count
      }

      mutating func append(media: LlamaPreparedInputUnit, endingAt positionCount: Int) {
        self.units.append(media)
        self.positionCount = positionCount
      }
    }

    private struct State: ~Copyable {
      var context: LlamaRuntimeContext?
      var cachedInputs = [Int: CachedInput]()
      var allocatedSequenceIds = Set<Int>()
      var logitsSequenceId: Int?

      borrowing func withContext<R: ~Copyable>(
        _ body: (borrowing LlamaRuntimeContext) throws -> R
      ) throws -> R {
        switch self.context {
        case .some(let context):
          return try body(context)
        case .none:
          throw LlamaRuntimeError(
            code: .contextCreationFailed,
            message: "The llama context is unavailable."
          )
        }
      }
    }

    let model: LlamaModelBox
    let parameters: LlamaContextParameters
    private let state = Lock(State())

    init(model: LlamaModelBox, parameters: LlamaContextParameters) {
      self.model = model
      self.parameters = parameters
    }

    func lease(copyingFrom parentSequenceId: Int?) -> LlamaSequenceLease? {
      self.state.withLock { state in
        guard
          let sequenceId = (0..<Int(self.parameters.maximumSequenceCount))
            .first(where: { !state.allocatedSequenceIds.contains($0) })
        else {
          return nil
        }
        state.allocatedSequenceIds.insert(sequenceId)
        if let parentSequenceId {
          if state.context != nil {
            let copied = try? state.withContext {
              $0.memoryCopy(
                source: parentSequenceId,
                destination: sequenceId,
                from: 0,
                to: -1
              )
            }
            if copied != true {
              _ = try? state.withContext {
                $0.memoryRemove(sequenceId: sequenceId, from: 0, to: -1)
              }
              state.allocatedSequenceIds.remove(sequenceId)
              return nil
            }
          }
          state.cachedInputs[sequenceId] = state.cachedInputs[parentSequenceId] ?? CachedInput()
        } else {
          state.cachedInputs[sequenceId] = CachedInput()
        }
        return LlamaSequenceLease(store: self, sequenceId: sequenceId)
      }
    }

    func synchronize(
      sequenceId: Int,
      input: borrowing LlamaPreparedInput,
      multimodalRuntime: LlamaMultimodalRuntime?,
      output: LlamaEvaluationOutput
    ) throws -> Int {
      try self.state.withLock { state in
        try self.ensureContext(&state)
        let cached = state.cachedInputs[sequenceId]?.units ?? []
        var prefixCount = zip(cached, input.units).prefix { $0 == $1 }.count
        if output.wantsLogits && prefixCount == input.units.count && !input.units.isEmpty {
          prefixCount = input.units.count - 1
        }
        prefixCount = self.trim(sequenceId: sequenceId, to: prefixCount, state: &state)
        if input.hasMedia {
          try self.evaluate(
            input,
            from: prefixCount,
            sequenceId: sequenceId,
            multimodalRuntime: multimodalRuntime,
            output: output,
            state: &state
          )
        } else {
          try self.evaluate(
            tokenIds: input.units[prefixCount...].tokenIds,
            sequenceId: sequenceId,
            output: output,
            state: &state
          )
        }
        guard !output.wantsLogits || state.logitsSequenceId == sequenceId else {
          throw LlamaRuntimeError(
            code: .decodeFailed,
            message: "The evaluated input produced no logits for its last position."
          )
        }
        return input.units[prefixCount...].reduce(0) { $0 + $1.tokenCount }
      }
    }

    func withLogits<R>(
      sequenceId: Int,
      appending pendingTokenId: EdgeToolsToken.ID?,
      vocabularySize: Int,
      _ body: (inout MutableSpan<Float>) throws -> R
    ) throws -> R {
      try self.state.withLock { state in
        try self.ensureContext(&state)
        if let pendingTokenId {
          try self.evaluate(
            tokenIds: CollectionOfOne(pendingTokenId),
            sequenceId: sequenceId,
            output: .lastTokenLogits,
            state: &state
          )
        } else if state.logitsSequenceId != sequenceId {
          let cached = state.cachedInputs[sequenceId]?.units ?? []
          guard case .token(let tokenId) = cached.last?.value else {
            throw LlamaRuntimeError(
              code: .decodeFailed,
              message: "The sequence has no evaluated tokens to produce logits from."
            )
          }
          let retained = self.trim(sequenceId: sequenceId, to: cached.count - 1, state: &state)
          guard retained == cached.count - 1 else {
            throw LlamaRuntimeError(
              code: .decodeFailed,
              message: "The sequence could not be rewound to recover its logits."
            )
          }
          try self.evaluate(
            tokenIds: CollectionOfOne(tokenId),
            sequenceId: sequenceId,
            output: .lastTokenLogits,
            state: &state
          )
        }
        guard let logits = try state.withContext({ $0.lastLogits() }) else {
          throw LlamaRuntimeError(
            code: .decodeFailed,
            message: "The llama context has no logits for the last evaluated token."
          )
        }
        let buffer = UnsafeMutableBufferPointer(start: logits, count: vocabularySize)
        var span = MutableSpan<Float>(_unsafeElements: buffer)
        return try body(&span)
      }
    }

    func commit(tokenId: EdgeToolsToken.ID, sequenceId: Int) throws {
      try self.state.withLock { state in
        try self.ensureContext(&state)
        try self.evaluate(
          tokenIds: CollectionOfOne(tokenId),
          sequenceId: sequenceId,
          output: .none,
          state: &state
        )
      }
    }

    func probeConfidence(sequenceId: Int) -> Float? {
      self.state.withLock { $0.context?.probeConfidence(sequenceId: sequenceId) ?? nil }
    }

    func resetProbe(sequenceId: Int) {
      self.state.withLock { $0.context?.probeReset(sequenceId: sequenceId) }
    }

    func warmUp() throws {
      try self.state.withLock { try self.ensureContext(&$0) }
    }

    fileprivate func release(sequenceId: Int) {
      self.state.withLock { state in
        _ = state.context?.memoryRemove(sequenceId: sequenceId, from: 0, to: -1)
        state.cachedInputs[sequenceId] = nil
        state.allocatedSequenceIds.remove(sequenceId)
        if state.logitsSequenceId == sequenceId {
          state.logitsSequenceId = nil
        }
      }
    }

    private func trim(sequenceId: Int, to prefixCount: Int, state: inout State) -> Int {
      let cached = state.cachedInputs[sequenceId] ?? CachedInput()
      guard prefixCount < cached.units.count else { return prefixCount }
      var prefixCount = prefixCount
      var retainedUnits = Array(cached.units.prefix(prefixCount))
      var retainedPositions = retainedUnits.reduce(0) { $0 + $1.positionCount }
      if state.context?.memoryRemove(sequenceId: sequenceId, from: retainedPositions, to: -1) != true {
        _ = state.context?.memoryRemove(sequenceId: sequenceId, from: 0, to: -1)
        prefixCount = 0
        retainedUnits = []
        retainedPositions = 0
      }
      state.cachedInputs[sequenceId] = CachedInput(
        units: retainedUnits,
        positionCount: retainedPositions
      )
      return prefixCount
    }

    private func evaluate(
      tokenIds: some Collection<EdgeToolsToken.ID>,
      sequenceId: Int,
      output: LlamaEvaluationOutput,
      state: inout State
    ) throws {
      let chunkSize = try state.withContext { $0.batchCapacity }
      var start = tokenIds.startIndex
      while start != tokenIds.endIndex {
        let end =
          tokenIds.index(
            start,
            offsetBy: chunkSize,
            limitedBy: tokenIds.endIndex
          ) ?? tokenIds.endIndex
        let chunk = tokenIds[start..<end]
        let startPosition = state.cachedInputs[sequenceId]?.positionCount ?? 0
        let wantsLogits = output.wantsLogits && end == tokenIds.endIndex
        try state.withContext {
          try $0.decode(
            tokenIds: chunk,
            startPosition: startPosition,
            sequenceId: sequenceId,
            wantsLogits: wantsLogits
          )
        }
        state.logitsSequenceId = wantsLogits ? sequenceId : nil
        state.cachedInputs[sequenceId, default: CachedInput()].append(tokenIds: chunk)
        start = end
      }
    }

    private func evaluate(
      _ input: borrowing LlamaPreparedInput,
      from prefixCount: Int,
      sequenceId: Int,
      multimodalRuntime: LlamaMultimodalRuntime?,
      output: LlamaEvaluationOutput,
      state: inout State
    ) throws {
      guard let multimodalRuntime else {
        throw LlamaRuntimeError(
          code: .multimodalProcessingFailed,
          message: "The multimodal runtime is unavailable."
        )
      }
      for chunk in input.chunks {
        switch chunk {
        case .text(let tokenIds, let units):
          guard prefixCount < units.upperBound else { continue }
          let offset = max(prefixCount - units.lowerBound, 0)
          try self.evaluate(
            tokenIds: tokenIds.dropFirst(offset),
            sequenceId: sequenceId,
            output: output.wantsLogits && units.upperBound == input.units.count
              ? .lastTokenLogits
              : .none,
            state: &state
          )
        case .media(let chunkIndex, let unitIndex):
          guard prefixCount <= unitIndex else { continue }
          let unit = input.units[unitIndex]
          let currentPosition = state.cachedInputs[sequenceId]?.positionCount ?? 0
          let batchSize = try state.withContext { $0.microBatchCapacity }
          let wantsLogits = output.wantsLogits && unitIndex == input.units.count - 1
          let newPosition = try state.withContext {
            try multimodalRuntime.evaluate(
              input: input,
              context: $0,
              chunkIndex: chunkIndex,
              position: currentPosition,
              sequenceId: sequenceId,
              batchSize: batchSize,
              wantsLogits: wantsLogits
            )
          }
          state.logitsSequenceId = wantsLogits ? sequenceId : nil
          state.cachedInputs[sequenceId, default: CachedInput()]
            .append(media: unit, endingAt: newPosition)
        }
      }
    }

    private func ensureContext(_ state: inout State) throws {
      guard state.context == nil else { return }
      state.context = try self.model.model.createContext(parameters: self.parameters)
    }
  }
#endif
