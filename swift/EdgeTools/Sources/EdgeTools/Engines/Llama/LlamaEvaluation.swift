#if Llama && canImport(CLlama)
  import EdgeToolsCore
  import EdgeToolsLlama

  // MARK: - LlamaEvaluationSegment

  enum LlamaEvaluationSegment {
    case tokens(ArraySlice<EdgeToolsToken.ID>)
    case media(chunkIndex: Int, unit: LlamaPreparedInputUnit)

    var tokenCount: Int {
      switch self {
      case .tokens(let tokenIds): tokenIds.count
      case .media(_, let unit): unit.tokenCount
      }
    }
  }

  // MARK: - LlamaEvaluation

  struct LlamaEvaluation {
    let segments: [LlamaEvaluationSegment]
    let tokenCount: Int

    init(input: borrowing LlamaPreparedInput, from prefixCount: Int) {
      self.init(input: input, units: prefixCount..<input.units.count)
    }

    init(input: borrowing LlamaPreparedInput, units range: Range<Int>) {
      var segments = [LlamaEvaluationSegment]()
      for chunk in input.chunks {
        switch chunk {
        case .text(let tokenIds, let units):
          let overlap = units.clamped(to: range)
          guard !overlap.isEmpty else { continue }
          let start = tokenIds.startIndex + (overlap.lowerBound - units.lowerBound)
          segments.append(.tokens(tokenIds[start..<(start + overlap.count)]))
        case .media(let chunkIndex, let unitIndex):
          guard range.contains(unitIndex) else { continue }
          segments.append(.media(chunkIndex: chunkIndex, unit: input.units[unitIndex]))
        }
      }
      self.segments = segments
      self.tokenCount = segments.reduce(0) { $0 + $1.tokenCount }
    }
  }

  // MARK: - LlamaLogitsEvaluation

  // The tail is resolved here rather than discovered while decoding. It is a single decode, so
  // the switch that evaluates it must produce `LlamaDecodedLogits` in every case; a loop's last
  // iteration could not be required to.
  struct LlamaLogitsEvaluation {
    enum Tail {
      case token(EdgeToolsToken.ID)
      case media(chunkIndex: Int, unit: LlamaPreparedInputUnit)

      var tokenCount: Int {
        switch self {
        case .token: 1
        case .media(_, let unit): unit.tokenCount
        }
      }
    }

    let leading: LlamaEvaluation
    let tail: Tail
    let tokenCount: Int

    init(input: borrowing LlamaPreparedInput, from prefixCount: Int) throws {
      guard let lastChunk = input.chunks.last else {
        throw LlamaRuntimeError(
          code: .decodeFailed,
          message: "The prepared input has nothing to evaluate for logits."
        )
      }
      let tail: Tail
      let tailUnitIndex: Int
      switch lastChunk {
      case .text(let tokenIds, let units):
        guard let tokenId = tokenIds.last else {
          throw LlamaRuntimeError(
            code: .decodeFailed,
            message: "The prepared input ends in an empty text chunk."
          )
        }
        tail = .token(tokenId)
        tailUnitIndex = units.upperBound - 1
      case .media(let chunkIndex, let unitIndex):
        tail = .media(chunkIndex: chunkIndex, unit: input.units[unitIndex])
        tailUnitIndex = unitIndex
      }
      let leading = LlamaEvaluation(
        input: input,
        units: min(prefixCount, tailUnitIndex)..<tailUnitIndex
      )
      self.leading = leading
      self.tail = tail
      self.tokenCount = leading.tokenCount + tail.tokenCount
    }
  }

  // MARK: - LlamaPreparedInput + Logits

  extension LlamaPreparedInput {
    // The largest cached prefix that still leaves a unit to decode for logits.
    var logitsRetainableUnitCount: Int {
      max(self.units.count - 1, 0)
    }
  }
#endif
