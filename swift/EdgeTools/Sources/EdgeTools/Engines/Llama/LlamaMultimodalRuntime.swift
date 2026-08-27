#if Llama && canImport(CLlama)
  import CLlama
  import EdgeToolsCore
  import EdgeToolsLlama
  import EdgeToolsTokenizers

  // MARK: - LlamaMultimodalParameters

  public struct LlamaMultimodalParameters: Hashable, Sendable {
    public var useGPU: Bool
    public var printTimings: Bool
    public var threadCount: Int32
    public var flashAttention: LlamaFlashAttention
    public var warmUp: Bool
    public var minimumImageTokenCount: Int32?
    public var maximumImageTokenCount: Int32?

    public init(
      useGPU: Bool = true,
      printTimings: Bool = false,
      threadCount: Int32 = 0,
      flashAttention: LlamaFlashAttention = .auto,
      warmUp: Bool = true,
      minimumImageTokenCount: Int32? = nil,
      maximumImageTokenCount: Int32? = nil
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

  enum LlamaMediaKind: Equatable, Sendable {
    case image
    case audio
  }

  struct LlamaMultimodalAsset: Sendable {
    let kind: LlamaMediaKind
    let asset: EdgeToolsTranscript.Asset
  }

  struct LlamaMultimodalInputProcessor<Profile: LlamaModelProfile>: Sendable {
    let tokenizer: LlamaTokenizer
    let runtime: LlamaMultimodalRuntime

    func input(
      prompt: EdgeToolsTranscript,
      reasoningEffort: EdgeToolsReasoningEffort,
      tools: [EdgeToolDefinition],
      addGenerationPrompt: Bool,
      kind: EdgeToolsLLMInputKind,
      cache: EdgeToolsLLMPreparedInputCache<LlamaPreparedInput>
    ) throws -> LlamaPreparedInput {
      guard prompt.videos.isEmpty else {
        throw EdgeToolsError.unsupportedMedia("Video input is not supported by LlamaEngine.")
      }
      guard let profile = Profile.self as? any EdgeToolsMultimodalModelProfile.Type else {
        throw EdgeToolsError.unsupportedTokenizer
      }
      var media = [LlamaMultimodalAsset]()
      let messages = try prompt.chatTemplateMessages { userMessage in
        let content = try profile.multimodalContent(for: userMessage)
          .reduce(into: "") { content, part in
            switch part {
            case .text(let text):
              content.append(text)
            case .image(let image):
              content.append(self.runtime.mediaMarker)
              media.append(LlamaMultimodalAsset(kind: .image, asset: image))
            case .audio(let audio):
              content.append(self.runtime.mediaMarker)
              media.append(LlamaMultimodalAsset(kind: .audio, asset: audio))
            case .video:
              throw EdgeToolsError.unsupportedMedia(
                "Video input is not supported by LlamaEngine."
              )
            }
          }
        return ["role": "user", "content": .string(content)]
      }
      let text = try self.tokenizer.renderChatTemplate(
        messages: messages,
        tools: tools.chatTemplateToolValues,
        addGenerationPrompt: addGenerationPrompt,
        additionalContext: Profile.templateContext(
          prompt: prompt,
          reasoningEffort: reasoningEffort
        )
      )
      let cachedInput = cache.input(
        for: prompt,
        reasoningEffort: reasoningEffort,
        tools: tools,
        kind: kind,
        allowingTextOnlyContinuation: true
      )
      if let cached = cachedInput,
        let input = try cached.replacingText(
          text,
          mediaMarker: self.runtime.mediaMarker,
          tokenizer: self.tokenizer
        )
      {
        cache.store(
          input,
          for: prompt,
          reasoningEffort: reasoningEffort,
          tools: tools,
          kind: kind
        )
        return input
      }
      let input = try self.runtime.prepare(text: text, media: media)
      cache.store(
        input,
        for: prompt,
        reasoningEffort: reasoningEffort,
        tools: tools,
        kind: kind
      )
      return input
    }
  }

  enum LlamaPreparedInputUnit: Equatable, Sendable {
    struct Media: Equatable, Sendable {
      let kind: LlamaMediaKind
      let id: String
      let chunkIndex: Int
      let tokenCount: Int
      let positionCount: Int
    }

    case token(EdgeToolsToken.ID)
    case media(Media)

    var tokenCount: Int {
      switch self {
      case .token: 1
      case .media(let media): media.tokenCount
      }
    }

    var positionCount: Int {
      switch self {
      case .token: 1
      case .media(let media): media.positionCount
      }
    }
  }

  // The handle is transferred once and all shared access is serialized by LlamaPreparedMedia.
  // Remove the unchecked conformance when imported C handles can express owned sendability.
  private struct LlamaInputChunks: ~Copyable, @unchecked Sendable {
    private let handle: OpaquePointer

    init() throws {
      guard let handle = mtmd_input_chunks_init() else {
        throw LlamaRuntimeError(
          code: .multimodalProcessingFailed,
          message: "The multimodal input chunks could not be allocated."
        )
      }
      self.handle = handle
    }

    deinit {
      mtmd_input_chunks_free(self.handle)
    }

    borrowing func tokenize(
      runtime: OpaquePointer,
      text: String,
      bitmapHandles: [OpaquePointer]
    ) -> Int32 {
      text.withCString { textPointer in
        var input = mtmd_input_text(
          text: textPointer,
          text_len: text.utf8.count,
          add_special: false,
          parse_special: true
        )
        var pointers = bitmapHandles.map(Optional.some)
        return pointers.withUnsafeMutableBufferPointer { pointers in
          mtmd_tokenize(
            runtime,
            self.handle,
            &input,
            pointers.baseAddress,
            pointers.count
          )
        }
      }
    }

    borrowing func units() throws -> [LlamaPreparedInputUnit] {
      var units = [LlamaPreparedInputUnit]()
      for chunkIndex in 0..<mtmd_input_chunks_size(self.handle) {
        guard let chunk = mtmd_input_chunks_get(self.handle, chunkIndex) else { continue }
        switch mtmd_input_chunk_get_type(chunk) {
        case MTMD_INPUT_CHUNK_TYPE_TEXT:
          var tokenCount = 0
          let tokens = mtmd_input_chunk_get_tokens_text(chunk, &tokenCount)
          let tokenIds = (0..<tokenCount).map { EdgeToolsToken.ID(tokens![$0]) }
          units.append(contentsOf: tokenIds.map(LlamaPreparedInputUnit.token))
        case MTMD_INPUT_CHUNK_TYPE_IMAGE, MTMD_INPUT_CHUNK_TYPE_AUDIO:
          let kind: LlamaMediaKind =
            mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_AUDIO ? .audio : .image
          let tokenCount = Int(mtmd_input_chunk_get_n_tokens(chunk))
          let positionCount = Int(mtmd_input_chunk_get_n_pos(chunk))
          let id = mtmd_input_chunk_get_id(chunk).map { String(cString: $0) } ?? ""
          units.append(
            .media(
              LlamaPreparedInputUnit.Media(
                kind: kind,
                id: id,
                chunkIndex: chunkIndex,
                tokenCount: tokenCount,
                positionCount: positionCount
              )
            )
          )
        default:
          throw EdgeToolsError.unsupportedMedia("Unsupported llama multimodal input chunk.")
        }
      }
      return units
    }

    borrowing func evaluate(
      runtime: OpaquePointer,
      context: borrowing LlamaRuntimeContext,
      chunkIndex: Int,
      position: Int,
      sequenceId: Int,
      batchSize: Int,
      wantsLogits: Bool
    ) throws -> Int {
      guard let chunk = mtmd_input_chunks_get(self.handle, chunkIndex) else {
        throw LlamaRuntimeError(
          code: .multimodalProcessingFailed,
          message: "The multimodal input chunk is unavailable."
        )
      }
      var newPosition = llama_pos(position)
      let status = mtmd_helper_eval_chunk_single(
        runtime,
        context.handle,
        chunk,
        llama_pos(position),
        llama_seq_id(sequenceId),
        Int32(clamping: batchSize),
        wantsLogits,
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

  private final class LlamaPreparedMedia: Sendable {
    private let chunks: Lock<LlamaInputChunks>

    init(chunks: consuming sending LlamaInputChunks) {
      self.chunks = Lock(chunks)
    }

    func evaluate(
      runtime: OpaquePointer,
      context: borrowing LlamaRuntimeContext,
      chunkIndex: Int,
      position: Int,
      sequenceId: Int,
      batchSize: Int,
      wantsLogits: Bool
    ) throws -> Int {
      try self.chunks.withBorrowedLock {
        try $0.evaluate(
          runtime: runtime,
          context: context,
          chunkIndex: chunkIndex,
          position: position,
          sequenceId: sequenceId,
          batchSize: batchSize,
          wantsLogits: wantsLogits
        )
      }
    }
  }

  struct LlamaPreparedInput: Sendable {
    private let media: LlamaPreparedMedia?
    let units: [LlamaPreparedInputUnit]

    init(tokenIds: [EdgeToolsToken.ID]) {
      self.media = nil
      self.units = tokenIds.map(LlamaPreparedInputUnit.token)
    }

    fileprivate init(chunks: consuming sending LlamaInputChunks) throws {
      self.units = try chunks.units()
      self.media = LlamaPreparedMedia(chunks: consume chunks)
    }

    private init(media: LlamaPreparedMedia, units: [LlamaPreparedInputUnit]) {
      self.media = media
      self.units = units
    }

    func replacingText(_ text: String, mediaMarker: String, tokenizer: LlamaTokenizer) throws -> Self? {
      guard let preparedMedia = self.media else { return nil }
      let mediaUnits = self.units.compactMap { unit -> LlamaPreparedInputUnit? in
        guard case .media = unit else { return nil }
        return unit
      }
      let segments = llamaTextSegments(text, separatedBy: mediaMarker)
      guard segments.count == mediaUnits.count + 1 else { return nil }

      var units = [LlamaPreparedInputUnit]()
      for (index, segment) in segments.enumerated() {
        let tokenIds = try tokenizer.tokenIds(text: String(segment), addSpecialTokens: false)
        units.append(contentsOf: tokenIds.map(LlamaPreparedInputUnit.token))
        if index < mediaUnits.count {
          units.append(mediaUnits[index])
        }
      }
      return Self(media: preparedMedia, units: units)
    }

    func evaluateMedia(
      runtime: OpaquePointer,
      context: borrowing LlamaRuntimeContext,
      chunkIndex: Int,
      position: Int,
      sequenceId: Int,
      batchSize: Int,
      wantsLogits: Bool
    ) throws -> Int {
      guard let media else {
        throw LlamaRuntimeError(
          code: .multimodalProcessingFailed,
          message: "The multimodal input chunk is unavailable."
        )
      }
      return try media.evaluate(
        runtime: runtime,
        context: context,
        chunkIndex: chunkIndex,
        position: position,
        sequenceId: sequenceId,
        batchSize: batchSize,
        wantsLogits: wantsLogits
      )
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
        contextParameters.n_threads = parameters.threadCount
      }
      contextParameters.flash_attn_type = llama_flash_attn_type(
        rawValue: parameters.flashAttention.rawValue
      )
      contextParameters.warmup = parameters.warmUp
      if let minimumImageTokenCount = parameters.minimumImageTokenCount {
        contextParameters.image_min_tokens = minimumImageTokenCount
      }
      if let maximumImageTokenCount = parameters.maximumImageTokenCount {
        contextParameters.image_max_tokens = maximumImageTokenCount
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
        let chunks = try LlamaInputChunks()
        let status = chunks.tokenize(
          runtime: state.handle,
          text: text,
          bitmapHandles: bitmapHandles
        )
        guard status == 0 else {
          throw LlamaRuntimeError(
            code: .multimodalProcessingFailed,
            message: "The multimodal prompt could not be tokenized (status \(status))."
          )
        }
        return try LlamaPreparedInput(chunks: consume chunks)
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
        try input.evaluateMedia(
          runtime: state.handle,
          context: context,
          chunkIndex: chunkIndex,
          position: position,
          sequenceId: sequenceId,
          batchSize: batchSize,
          wantsLogits: false
        )
      }
    }

    func evaluateProducingLogits(
      input: borrowing LlamaPreparedInput,
      context: borrowing LlamaRuntimeContext,
      chunkIndex: Int,
      position: Int,
      sequenceId: Int,
      batchSize: Int
    ) throws -> (position: Int, logits: LlamaDecodedLogits) {
      let position = try self.state.withBorrowedLock { state in
        try input.evaluateMedia(
          runtime: state.handle,
          context: context,
          chunkIndex: chunkIndex,
          position: position,
          sequenceId: sequenceId,
          batchSize: batchSize,
          wantsLogits: true
        )
      }
      return (position, LlamaDecodedLogits(sequenceId: sequenceId))
    }
  }

  // MARK: - Helpers

  extension Sequence<LlamaPreparedInputUnit> {
    var tokenIds: [EdgeToolsToken.ID] {
      self.compactMap { unit in
        guard case .token(let tokenId) = unit else { return nil }
        return tokenId
      }
    }
  }

  private func llamaTextSegments(_ text: String, separatedBy separator: String) -> [Substring] {
    guard !separator.isEmpty else { return [text[...]] }
    var segments = [Substring]()
    var remainder = text[...]
    while let range = remainder.firstRange(of: separator) {
      segments.append(remainder[..<range.lowerBound])
      remainder = remainder[range.upperBound...]
    }
    segments.append(remainder)
    return segments
  }
#endif
