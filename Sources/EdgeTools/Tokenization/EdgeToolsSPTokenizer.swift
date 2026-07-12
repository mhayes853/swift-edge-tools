#if Sentencepiece
  import CSentencepiece
  import Foundation

  public struct EdgeToolsSPTokenizer: ~Copyable {
    private let tokenizer: sp_tokenizer_t

    public var vocabularySize: Int {
      Int(sp_tokenizer_vocab_size(self.tokenizer))
    }

    public var bosTokenId: EdgeToolsToken.ID? {
      Self.optionalTokenID(sp_tokenizer_bos_token_id(self.tokenizer))
    }

    public var eosTokenId: EdgeToolsToken.ID? {
      Self.optionalTokenID(sp_tokenizer_eos_token_id(self.tokenizer))
    }

    public var unknownTokenId: EdgeToolsToken.ID? {
      Self.optionalTokenID(sp_tokenizer_unk_token_id(self.tokenizer))
    }

    public var padTokenId: EdgeToolsToken.ID? {
      Self.optionalTokenID(sp_tokenizer_pad_token_id(self.tokenizer))
    }

    public init(modelURL: URL) throws {
      let nativePath = modelURL.withUnsafeFileSystemRepresentation { String(cString: $0!) }
      guard let tokenizer = sp_tokenizer_init_from_file(nativePath) else {
        throw EdgeToolsSPTokenizerError()
      }
      self.init(tokenizer: tokenizer)
    }

    public init(data: Data) throws {
      let tokenizer = try data.withUnsafeBytes { rawBuffer in
        guard let base = rawBuffer.baseAddress else { throw EdgeToolsSPTokenizerError() }
        let tokenizer = sp_tokenizer_init_from_data(
          base.assumingMemoryBound(to: CChar.self),
          rawBuffer.count
        )
        guard let tokenizer else { throw EdgeToolsSPTokenizerError() }
        return tokenizer
      }
      self.init(tokenizer: tokenizer)
    }

    init(tokenizer: consuming sp_tokenizer_t) {
      self.tokenizer = tokenizer
    }

    deinit {
      sp_tokenizer_destroy(self.tokenizer)
    }

    public func encode(text: String) -> [EdgeToolsToken.ID] {
      var size = 0
      let tokenIds = sp_tokenizer_encode(self.tokenizer, text, &size)
      defer { tokenIds?.deallocate() }
      return Array(UnsafeBufferPointer(start: tokenIds, count: Int(size))).map(Int.init)
    }

    public func decode(tokens: [EdgeToolsToken.ID]) -> String {
      let tokenIds = tokens.map(Int32.init)
      let string = tokenIds.withUnsafeBufferPointer { tokenIdsPtr in
        sp_tokenizer_decode(self.tokenizer, tokenIdsPtr.baseAddress, tokenIdsPtr.count, nil)
      }
      defer { string?.deallocate() }
      return string.map { String(cString: $0) } ?? ""
    }

    public func convertTokensToIds(_ tokens: [String]) -> [EdgeToolsToken.ID?] {
      let unknownTokenId = self.unknownTokenId
      return tokens.map { token in
        let tokenId = Int(token.withCString { sp_tokenizer_token_to_id(self.tokenizer, $0) })
        return token == self.unknownToken || tokenId != unknownTokenId ? tokenId : nil
      }
    }

    public func convertIdsToTokens(_ ids: [EdgeToolsToken.ID]) -> [String?] {
      ids.map { id in
        withUnsafeTemporaryAllocation(of: CChar.self, capacity: 256) { buffer in
          let result = sp_tokenizer_id_to_token(
            self.tokenizer,
            Int32(id),
            buffer.baseAddress,
            buffer.count
          )
          return result == 0 ? String(cString: buffer.baseAddress!) : nil
        }
      }
    }

    private static func optionalTokenID(_ tokenID: Int32) -> EdgeToolsToken.ID? {
      let tokenID = Int(tokenID)
      return tokenID >= 0 ? tokenID : nil
    }
  }

  extension EdgeToolsSPTokenizer: EdgeToolsTokenizer {}

  public struct EdgeToolsSPTokenizerError: Hashable, Sendable, Error {
    public let message: String

    init() {
      self.message = String(cString: sp_tokenizer_last_error_message())
    }
  }
#endif
