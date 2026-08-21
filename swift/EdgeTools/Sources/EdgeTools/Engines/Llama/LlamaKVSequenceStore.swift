#if Llama && canImport(CLlama)
  import EdgeToolsCore
  import EdgeToolsLlama

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
      multimodalRuntime: LlamaMultimodalRuntime?
    ) throws -> Int {
      try self.state.withLock { state in
        try self.ensureContext(&state)
        let prefixCount = self.prefix(
          matching: input,
          sequenceId: sequenceId,
          retaining: input.units.count,
          state: &state
        )
        let evaluation = LlamaEvaluation(input: input, from: prefixCount)
        try self.evaluate(
          evaluation,
          input: input,
          sequenceId: sequenceId,
          multimodalRuntime: multimodalRuntime,
          state: &state
        )
        return evaluation.tokenCount
      }
    }

    func synchronizeForLogits(
      sequenceId: Int,
      input: borrowing LlamaPreparedInput,
      multimodalRuntime: LlamaMultimodalRuntime?
    ) throws -> Int {
      try self.state.withLock { state in
        try self.ensureContext(&state)
        let prefixCount = self.prefix(
          matching: input,
          sequenceId: sequenceId,
          retaining: input.logitsRetainableUnitCount,
          state: &state
        )
        let evaluation = try LlamaLogitsEvaluation(input: input, from: prefixCount)
        try self.evaluate(
          evaluation.leading,
          input: input,
          sequenceId: sequenceId,
          multimodalRuntime: multimodalRuntime,
          state: &state
        )
        let logits = try self.evaluate(
          tail: evaluation.tail,
          input: input,
          sequenceId: sequenceId,
          multimodalRuntime: multimodalRuntime,
          state: &state
        )
        state.logitsSequenceId = logits.sequenceId
        return evaluation.tokenCount
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
          let logits = try self.decodeProducingLogits(
            tokenId: pendingTokenId,
            sequenceId: sequenceId,
            state: &state
          )
          state.logitsSequenceId = logits.sequenceId
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
          let logits = try self.decodeProducingLogits(
            tokenId: tokenId,
            sequenceId: sequenceId,
            state: &state
          )
          state.logitsSequenceId = logits.sequenceId
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
        try self.decode(tokenIds: CollectionOfOne(tokenId), sequenceId: sequenceId, state: &state)
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

    private func prefix(
      matching input: borrowing LlamaPreparedInput,
      sequenceId: Int,
      retaining retainableCount: Int,
      state: inout State
    ) -> Int {
      let cached = state.cachedInputs[sequenceId]?.units ?? []
      let matched = zip(cached, input.units).prefix { $0 == $1 }.count
      return self.trim(sequenceId: sequenceId, to: min(matched, retainableCount), state: &state)
    }

    private func evaluate(
      _ evaluation: borrowing LlamaEvaluation,
      input: borrowing LlamaPreparedInput,
      sequenceId: Int,
      multimodalRuntime: LlamaMultimodalRuntime?,
      state: inout State
    ) throws {
      for segment in evaluation.segments {
        switch segment {
        case .tokens(let tokenIds):
          try self.decode(tokenIds: tokenIds, sequenceId: sequenceId, state: &state)
        case .media(let chunkIndex, let unit):
          try self.decodeMedia(
            chunkIndex: chunkIndex,
            unit: unit,
            input: input,
            sequenceId: sequenceId,
            multimodalRuntime: multimodalRuntime,
            state: &state
          )
        }
      }
    }

    private func evaluate(
      tail: LlamaLogitsEvaluation.Tail,
      input: borrowing LlamaPreparedInput,
      sequenceId: Int,
      multimodalRuntime: LlamaMultimodalRuntime?,
      state: inout State
    ) throws -> LlamaDecodedLogits {
      switch tail {
      case .token(let tokenId):
        try self.decodeProducingLogits(tokenId: tokenId, sequenceId: sequenceId, state: &state)
      case .media(let chunkIndex, let unit):
        try self.decodeMediaProducingLogits(
          chunkIndex: chunkIndex,
          unit: unit,
          input: input,
          sequenceId: sequenceId,
          multimodalRuntime: multimodalRuntime,
          state: &state
        )
      }
    }

    private func decode(
      tokenIds: some Collection<EdgeToolsToken.ID>,
      sequenceId: Int,
      state: inout State
    ) throws {
      let capacity = try state.withContext { $0.batchCapacity }
      var start = tokenIds.startIndex
      while start != tokenIds.endIndex {
        let end =
          tokenIds.index(
            start,
            offsetBy: capacity,
            limitedBy: tokenIds.endIndex
          ) ?? tokenIds.endIndex
        let batch = tokenIds[start..<end]
        let startPosition = state.cachedInputs[sequenceId]?.positionCount ?? 0
        try state.withContext {
          try $0.decode(tokenIds: batch, startPosition: startPosition, sequenceId: sequenceId)
        }
        state.logitsSequenceId = nil
        state.cachedInputs[sequenceId, default: CachedInput()].append(tokenIds: batch)
        start = end
      }
    }

    private func decodeProducingLogits(
      tokenId: EdgeToolsToken.ID,
      sequenceId: Int,
      state: inout State
    ) throws -> LlamaDecodedLogits {
      let startPosition = state.cachedInputs[sequenceId]?.positionCount ?? 0
      let logits = try state.withContext {
        try $0.decodeProducingLogits(
          tokenIds: CollectionOfOne(tokenId),
          startPosition: startPosition,
          sequenceId: sequenceId
        )
      }
      state.cachedInputs[sequenceId, default: CachedInput()]
        .append(tokenIds: CollectionOfOne(tokenId))
      return logits
    }

    private func decodeMedia(
      chunkIndex: Int,
      unit: LlamaPreparedInputUnit,
      input: borrowing LlamaPreparedInput,
      sequenceId: Int,
      multimodalRuntime: LlamaMultimodalRuntime?,
      state: inout State
    ) throws {
      let runtime = try self.runtime(multimodalRuntime)
      let position = state.cachedInputs[sequenceId]?.positionCount ?? 0
      let batchSize = try state.withContext { $0.microBatchCapacity }
      let newPosition = try state.withContext {
        try runtime.evaluate(
          input: input,
          context: $0,
          chunkIndex: chunkIndex,
          position: position,
          sequenceId: sequenceId,
          batchSize: batchSize
        )
      }
      state.logitsSequenceId = nil
      state.cachedInputs[sequenceId, default: CachedInput()]
        .append(media: unit, endingAt: newPosition)
    }

    private func decodeMediaProducingLogits(
      chunkIndex: Int,
      unit: LlamaPreparedInputUnit,
      input: borrowing LlamaPreparedInput,
      sequenceId: Int,
      multimodalRuntime: LlamaMultimodalRuntime?,
      state: inout State
    ) throws -> LlamaDecodedLogits {
      let runtime = try self.runtime(multimodalRuntime)
      let position = state.cachedInputs[sequenceId]?.positionCount ?? 0
      let batchSize = try state.withContext { $0.microBatchCapacity }
      let evaluated = try state.withContext {
        try runtime.evaluateProducingLogits(
          input: input,
          context: $0,
          chunkIndex: chunkIndex,
          position: position,
          sequenceId: sequenceId,
          batchSize: batchSize
        )
      }
      state.cachedInputs[sequenceId, default: CachedInput()]
        .append(media: unit, endingAt: evaluated.position)
      return evaluated.logits
    }

    private func runtime(
      _ multimodalRuntime: LlamaMultimodalRuntime?
    ) throws -> LlamaMultimodalRuntime {
      guard let multimodalRuntime else {
        throw LlamaRuntimeError(
          code: .multimodalProcessingFailed,
          message: "The multimodal runtime is unavailable."
        )
      }
      return multimodalRuntime
    }

    private func ensureContext(_ state: inout State) throws {
      guard state.context == nil else { return }
      state.context = try self.model.model.createContext(parameters: self.parameters)
    }
  }
#endif
