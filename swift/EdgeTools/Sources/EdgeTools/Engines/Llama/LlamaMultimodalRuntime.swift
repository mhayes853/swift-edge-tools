#if Llama && canImport(CLlama)
  import CLlama
  import EdgeToolsCore
  import EdgeToolsLlama

  // MARK: - LlamaMultimodalParameters

  public struct LlamaMultimodalParameters: Hashable, Sendable {
    public var useGPU: Bool
    public var printTimings: Bool
    public var threadCount: Int
    public var flashAttention: LlamaFlashAttention
    public var warmUp: Bool
    public var minimumImageTokenCount: Int?
    public var maximumImageTokenCount: Int?

    public init(
      useGPU: Bool = true,
      printTimings: Bool = false,
      threadCount: Int = 0,
      flashAttention: LlamaFlashAttention = .auto,
      warmUp: Bool = true,
      minimumImageTokenCount: Int? = nil,
      maximumImageTokenCount: Int? = nil
    ) {
      self.useGPU = useGPU
      self.printTimings = printTimings
      self.threadCount = threadCount
      self.flashAttention = flashAttention
      self.warmUp = warmUp
      self.minimumImageTokenCount = minimumImageTokenCount
      self.maximumImageTokenCount = maximumImageTokenCount
    }
  }

  // MARK: - LlamaPreparedInput

  enum LlamaMediaKind: Equatable {
    case image
    case audio
  }

  struct LlamaMultimodalAsset {
    let kind: LlamaMediaKind
    let asset: EdgeToolsTranscript.Asset
  }

  struct LlamaPreparedInputUnit: Equatable {
    struct Media: Equatable {
      let kind: LlamaMediaKind
      let id: String
      let tokenCount: Int
      let positionCount: Int
    }

    enum Value: Equatable {
      case token(EdgeToolsToken.ID)
      case media(Media)
    }

    let value: Value

    var tokenCount: Int {
      switch self.value {
      case .token: 1
      case .media(let media): media.tokenCount
      }
    }

    var positionCount: Int {
      switch self.value {
      case .token: 1
      case .media(let media): media.positionCount
      }
    }
  }

  struct LlamaPreparedInput: ~Copyable, @unchecked Sendable {
    enum Chunk {
      case text(tokenIds: [EdgeToolsToken.ID], units: Range<Int>)
      case media(chunkIndex: Int, unit: Int)
    }

    let handle: OpaquePointer?
    let units: [LlamaPreparedInputUnit]
    let chunks: [Chunk]

    init(tokenIds: [EdgeToolsToken.ID]) {
      self.handle = nil
      self.units = tokenIds.map { LlamaPreparedInputUnit(value: .token($0)) }
      self.chunks = []
    }

    init(handle: consuming OpaquePointer) throws {
      var units = [LlamaPreparedInputUnit]()
      var chunks = [Chunk]()
      for chunkIndex in 0..<mtmd_input_chunks_size(handle) {
        guard let chunk = mtmd_input_chunks_get(handle, chunkIndex) else { continue }
        let unitStart = units.count
        switch mtmd_input_chunk_get_type(chunk) {
        case MTMD_INPUT_CHUNK_TYPE_TEXT:
          var tokenCount = 0
          let tokens = mtmd_input_chunk_get_tokens_text(chunk, &tokenCount)
          let tokenIds = (0..<tokenCount).map { EdgeToolsToken.ID(tokens![$0]) }
          units.append(contentsOf: tokenIds.map { LlamaPreparedInputUnit(value: .token($0)) })
          chunks.append(.text(tokenIds: tokenIds, units: unitStart..<units.count))
        case MTMD_INPUT_CHUNK_TYPE_IMAGE, MTMD_INPUT_CHUNK_TYPE_AUDIO:
          let kind: LlamaMediaKind =
            mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_AUDIO ? .audio : .image
          let tokenCount = Int(mtmd_input_chunk_get_n_tokens(chunk))
          let positionCount = Int(mtmd_input_chunk_get_n_pos(chunk))
          let id = mtmd_input_chunk_get_id(chunk).map { String(cString: $0) } ?? ""
          units.append(
            LlamaPreparedInputUnit(
              value: .media(
                LlamaPreparedInputUnit.Media(
                  kind: kind,
                  id: id,
                  tokenCount: tokenCount,
                  positionCount: positionCount
                )
              )
            )
          )
          chunks.append(.media(chunkIndex: chunkIndex, unit: unitStart))
        default:
          throw EdgeToolsError.unsupportedMedia("Unsupported llama multimodal input chunk.")
        }
      }
      self.handle = consume handle
      self.units = units
      self.chunks = chunks
    }

    deinit {
      if let handle = self.handle {
        mtmd_input_chunks_free(handle)
      }
    }
  }

  // MARK: - LlamaMultimodalRuntime

  final class LlamaMultimodalRuntime: Sendable {
    private struct State: ~Copyable {
      let handle: OpaquePointer

      deinit {
        mtmd_free(self.handle)
      }
    }

    private let model: LlamaModelBox
    private let state: Lock<State>

    init(
      path: String,
      model: LlamaModelBox,
      parameters: LlamaMultimodalParameters
    ) throws {
      var contextParameters = mtmd_context_params_default()
      contextParameters.use_gpu = parameters.useGPU
      contextParameters.print_timings = parameters.printTimings
      if parameters.threadCount > 0 {
        contextParameters.n_threads = Int32(clamping: parameters.threadCount)
      }
      contextParameters.flash_attn_type = llama_flash_attn_type(
        rawValue: parameters.flashAttention.rawValue
      )
      contextParameters.warmup = parameters.warmUp
      if let minimumImageTokenCount = parameters.minimumImageTokenCount {
        contextParameters.image_min_tokens = Int32(clamping: minimumImageTokenCount)
      }
      if let maximumImageTokenCount = parameters.maximumImageTokenCount {
        contextParameters.image_max_tokens = Int32(clamping: maximumImageTokenCount)
      }
      let handle = path.withCString { path in
        mtmd_init_from_file(path, model.model.handle, contextParameters)
      }
      guard let handle else {
        throw LlamaRuntimeError(
          code: .multimodalProjectorLoadFailed,
          message: "The multimodal projector at \(path) could not be loaded."
        )
      }
      guard mtmd_support_vision(handle) || mtmd_support_audio(handle) else {
        mtmd_free(handle)
        throw EdgeToolsError.unsupportedMedia(
          "The multimodal projector does not support image or audio input."
        )
      }
      self.model = model
      self.state = Lock(State(handle: handle))
    }

    var mediaMarker: String {
      self.state.withBorrowedLock { state in
        String(cString: mtmd_get_marker(state.handle))
      }
    }

    func prepare(text: String, media: [LlamaMultimodalAsset]) throws -> LlamaPreparedInput {
      try self.state.withBorrowedLock { state in
        var bitmapHandles = [OpaquePointer]()
        defer { bitmapHandles.forEach(mtmd_bitmap_free) }
        for item in media {
          switch item.kind {
          case .image:
            guard mtmd_support_vision(state.handle) else {
              throw EdgeToolsError.unsupportedMedia(
                "The multimodal projector does not support image input."
              )
            }
          case .audio:
            guard mtmd_support_audio(state.handle) else {
              throw EdgeToolsError.unsupportedMedia(
                "The multimodal projector does not support audio input."
              )
            }
          }
          let wrapper =
            switch item.asset.content {
            case .path(let path):
              path.withCString { mtmd_helper_bitmap_init_from_file(state.handle, $0, false) }
            case .bytes(let bytes):
              bytes.withUnsafeBytes { bytes in
                mtmd_helper_bitmap_init_from_buf(
                  state.handle,
                  bytes.bindMemory(to: UInt8.self).baseAddress,
                  bytes.count,
                  false
                )
              }
            }
          if let videoHandle = wrapper.video_ctx {
            mtmd_helper_video_free(videoHandle)
            wrapper.bitmap.map(mtmd_bitmap_free)
            throw EdgeToolsError.unsupportedMedia("Video input is not supported by LlamaEngine.")
          }
          guard let bitmapHandle = wrapper.bitmap else {
            let name = item.kind == .audio ? "audio" : "image"
            throw EdgeToolsError.invalidMedia("The \(name) could not be decoded.")
          }
          let decodedKind: LlamaMediaKind = mtmd_bitmap_is_audio(bitmapHandle) ? .audio : .image
          guard decodedKind == item.kind else {
            mtmd_bitmap_free(bitmapHandle)
            throw EdgeToolsError.invalidMedia(
              "The decoded media does not match its declared modality."
            )
          }
          bitmapHandles.append(bitmapHandle)
        }
        guard let chunks = mtmd_input_chunks_init() else {
          throw LlamaRuntimeError(
            code: .multimodalProcessingFailed,
            message: "The multimodal input chunks could not be allocated."
          )
        }
        do {
          let status = text.withCString { textPointer in
            var input = mtmd_input_text(
              text: textPointer,
              text_len: text.utf8.count,
              add_special: false,
              parse_special: true
            )
            var pointers = bitmapHandles.map(Optional.some)
            return pointers.withUnsafeMutableBufferPointer { pointers in
              mtmd_tokenize(
                state.handle,
                chunks,
                &input,
                pointers.baseAddress,
                pointers.count
              )
            }
          }
          guard status == 0 else {
            throw LlamaRuntimeError(
              code: .multimodalProcessingFailed,
              message: "The multimodal prompt could not be tokenized (status \(status))."
            )
          }
          return try LlamaPreparedInput(handle: chunks)
        } catch {
          mtmd_input_chunks_free(chunks)
          throw error
        }
      }
    }

    func evaluate(
      input: borrowing LlamaPreparedInput,
      context: borrowing LlamaRuntimeContext,
      chunkIndex: Int,
      position: Int,
      sequenceId: Int,
      batchSize: Int
    ) throws -> Int {
      try self.state.withBorrowedLock { state in
        guard
          let inputHandle = input.handle,
          let chunk = mtmd_input_chunks_get(inputHandle, chunkIndex)
        else {
          throw LlamaRuntimeError(
            code: .multimodalProcessingFailed,
            message: "The multimodal input chunk is unavailable."
          )
        }
        var newPosition = llama_pos(position)
        let status = mtmd_helper_eval_chunk_single(
          state.handle,
          context.handle,
          chunk,
          llama_pos(position),
          llama_seq_id(sequenceId),
          Int32(clamping: batchSize),
          false,
          &newPosition
        )
        guard status == 0 else {
          throw LlamaRuntimeError(
            code: .decodeFailed,
            message: "The multimodal chunk could not be evaluated (status \(status))."
          )
        }
        return Int(newPosition)
      }
    }
  }

  // MARK: - Helpers

  extension Sequence<LlamaPreparedInputUnit> {
    var tokenIds: [EdgeToolsToken.ID] {
      self.compactMap { unit in
        guard case .token(let tokenId) = unit.value else { return nil }
        return tokenId
      }
    }
  }
#endif
