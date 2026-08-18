#if Llama && canImport(CLlama)
  import EdgeToolsCore

  // MARK: - LlamaSequenceFamily

  final class LlamaSequenceFamily: Sendable {
    final class Lease: Sendable {
      let family: LlamaSequenceFamily
      let sequenceId: Int

      init(family: LlamaSequenceFamily, sequenceId: Int) {
        self.family = family
        self.sequenceId = sequenceId
      }

      deinit { self.family.release(sequenceId: self.sequenceId) }
    }

    private struct CachedInput {
      var units = [LlamaPreparedInputUnit]()
      var positionCount = 0

      mutating func append(tokenIds: [EdgeToolsToken.ID]) {
        self.units.append(contentsOf: tokenIds.map { LlamaPreparedInputUnit(value: .token($0)) })
        self.positionCount += tokenIds.count
      }

      mutating func append(media: LlamaPreparedInputUnit, endingAt positionCount: Int) {
        self.units.append(media)
        self.positionCount = positionCount
      }
    }

    private struct State: ~Copyable {
      var context: LlamaContextHandle?
      var cachedInputs = [Int: CachedInput]()
      var allocatedSequenceIds = Set<Int>()
      var logitsSequenceId: Int?

      borrowing func withContext<R: ~Copyable>(
        _ body: (borrowing LlamaContextHandle) throws -> R
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

    private static let decodeChunkSize = 512

    let model: LlamaModelBox
    let parameters: LlamaContextParameters
    let multimodalProjector: LlamaMultimodalProjector?
    private let state = Lock(State())

    init(
      model: LlamaModelBox,
      parameters: LlamaContextParameters,
      multimodalProjector: LlamaMultimodalProjector?
    ) {
      self.model = model
      self.parameters = parameters
      self.multimodalProjector = multimodalProjector
    }

    func lease(copyingFrom parentSequenceId: Int?) -> Lease? {
      self.state.withLock { state in
        guard
          let sequenceId = (0..<self.parameters.maximumSequenceCount)
            .first(where: { !state.allocatedSequenceIds.contains($0) })
        else {
          return nil
        }
        state.allocatedSequenceIds.insert(sequenceId)
        if let parentSequenceId {
          state.context?.memoryCopy(
            source: parentSequenceId,
            destination: sequenceId,
            from: 0,
            to: -1
          )
          state.cachedInputs[sequenceId] = state.cachedInputs[parentSequenceId] ?? CachedInput()
        } else {
          state.cachedInputs[sequenceId] = CachedInput()
        }
        return Lease(family: self, sequenceId: sequenceId)
      }
    }

    func prefill(
      sequenceId: Int,
      input: borrowing LlamaPreparedInput,
      wantsLogits: Bool
    ) throws -> Int {
      try self.state.withLock { state in
        try self.ensureContext(&state)
        let cached = state.cachedInputs[sequenceId]?.units ?? []
        var prefixCount = zip(cached, input.units).prefix { $0 == $1 }.count
        if wantsLogits && prefixCount == input.units.count && !input.units.isEmpty {
          prefixCount = input.units.count - 1
        }
        prefixCount = self.trim(sequenceId: sequenceId, to: prefixCount, state: &state)
        if input.handle != nil {
          try self.evaluate(
            input,
            from: prefixCount,
            sequenceId: sequenceId,
            wantsLogits: wantsLogits,
            state: &state
          )
        } else {
          try self.decode(
            tokenIds: input.units[prefixCount...].tokenIds,
            sequenceId: sequenceId,
            wantsLogits: wantsLogits,
            state: &state
          )
        }
        return input.units[prefixCount...].reduce(0) { $0 + $1.tokenCount }
      }
    }

    func withCurrentLogits<R>(
      sequenceId: Int,
      appending pendingTokenId: EdgeToolsToken.ID?,
      vocabularySize: Int,
      _ body: (inout MutableSpan<Float>) throws -> R
    ) throws -> R {
      try self.state.withLock { state in
        try self.ensureContext(&state)
        if let pendingTokenId {
          try self.decode(
            tokenIds: [pendingTokenId],
            sequenceId: sequenceId,
            wantsLogits: true,
            state: &state
          )
        } else if state.logitsSequenceId != sequenceId {
          let cached = state.cachedInputs[sequenceId]?.units ?? []
          guard case .token(let tokenId) = cached.last?.value else {
            throw LlamaRuntimeError(
              code: .decodeFailed,
              message: "The sequence has no decoded tokens to produce logits from."
            )
          }
          _ = self.trim(sequenceId: sequenceId, to: cached.count - 1, state: &state)
          try self.decode(
            tokenIds: [tokenId],
            sequenceId: sequenceId,
            wantsLogits: true,
            state: &state
          )
        }
        guard let logits = try state.withContext({ $0.lastLogits() }) else {
          throw LlamaRuntimeError(
            code: .decodeFailed,
            message: "The llama context has no logits for the last decoded token."
          )
        }
        var span = MutableSpan<Float>(
          _unsafeElements: UnsafeMutableBufferPointer(start: logits, count: vocabularySize)
        )
        return try body(&span)
      }
    }

    func append(tokenId: EdgeToolsToken.ID, sequenceId: Int) throws {
      try self.state.withLock { state in
        try self.ensureContext(&state)
        try self.decode(
          tokenIds: [tokenId],
          sequenceId: sequenceId,
          wantsLogits: false,
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

    private func release(sequenceId: Int) {
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
      if state.context?.memoryRemove(
        sequenceId: sequenceId,
        from: retainedPositions,
        to: -1
      ) != true {
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

    private func decode(
      tokenIds: [EdgeToolsToken.ID],
      sequenceId: Int,
      wantsLogits: Bool,
      state: inout State
    ) throws {
      for start in stride(from: 0, to: tokenIds.count, by: Self.decodeChunkSize) {
        let end = min(start + Self.decodeChunkSize, tokenIds.count)
        let chunk = Array(tokenIds[start..<end])
        let startPosition = state.cachedInputs[sequenceId]?.positionCount ?? 0
        try state.withContext {
          try $0.decode(
            tokenIds: chunk,
            startPosition: startPosition,
            sequenceId: sequenceId,
            wantsLogits: wantsLogits && end == tokenIds.count
          )
        }
        state.cachedInputs[sequenceId, default: CachedInput()].append(tokenIds: chunk)
      }
      if wantsLogits {
        state.logitsSequenceId = sequenceId
      }
    }

    private func evaluate(
      _ input: borrowing LlamaPreparedInput,
      from prefixCount: Int,
      sequenceId: Int,
      wantsLogits: Bool,
      state: inout State
    ) throws {
      guard let projector = self.multimodalProjector else {
        throw LlamaRuntimeError(
          code: .multimodalProcessingFailed,
          message: "The multimodal projector is unavailable."
        )
      }
      for chunk in input.chunks {
        switch chunk {
        case .text(let tokenIds, let units):
          guard prefixCount < units.upperBound else { continue }
          let offset = max(prefixCount - units.lowerBound, 0)
          try self.decode(
            tokenIds: Array(tokenIds.dropFirst(offset)),
            sequenceId: sequenceId,
            wantsLogits: wantsLogits && units.upperBound == input.units.count,
            state: &state
          )
        case .media(let chunkIndex, let unitIndex):
          guard prefixCount <= unitIndex else { continue }
          let unit = input.units[unitIndex]
          let currentPosition = state.cachedInputs[sequenceId]?.positionCount ?? 0
          let newPosition = try state.withContext {
            try projector.evaluate(
              input: input,
              context: $0,
              chunkIndex: chunkIndex,
              position: currentPosition,
              sequenceId: sequenceId,
              batchSize: Self.decodeChunkSize,
              wantsLogits: false
            )
          }
          state.cachedInputs[sequenceId, default: CachedInput()]
            .append(media: unit, endingAt: newPosition)
        }
      }
    }

    func warmUp() throws {
      try self.state.withLock { state in
        try self.ensureContext(&state)
      }
    }

    private func ensureContext(_ state: inout State) throws {
      guard state.context == nil else { return }
      state.context = try self.model.model.createContext(parameters: self.parameters)
    }
  }
#endif
