#if SwiftNeedleSentencepiece
  import Foundation
  import CNeedleSentencepiece

  // MARK: - NeedleSentencepieceTokenizer

  public struct NeedleSentencepieceTokenizer: ~Copyable {
    public let tokenizer: needle_sp_t

    public var unkTokenId: NeedleToken.ID { Int(needle_sp_unk_token_id(self.tokenizer)) }
    public var bosTokenId: NeedleToken.ID { Int(needle_sp_bos_token_id(self.tokenizer)) }
    public var eosTokenId: NeedleToken.ID { Int(needle_sp_eos_token_id(self.tokenizer)) }
    public var padTokenId: NeedleToken.ID { Int(needle_sp_pad_token_id(self.tokenizer)) }

    public init(modelURL: URL) throws {
      let nativePath = modelURL.withUnsafeFileSystemRepresentation { $0 }
      guard let tokenizer = needle_sp_init_from_file(nativePath) else {
        throw NeedleSentencepieceTokenizerError()
      }
      self.tokenizer = tokenizer
    }

    public init(tokenizer: consuming needle_sp_t) {
      self.tokenizer = tokenizer
    }

    deinit { needle_sp_destroy(self.tokenizer) }

    public func tokenIds(from tokens: [String]) -> [NeedleToken.ID] {
      let buffer = UnsafeMutablePointer<Int32>.allocate(capacity: tokens.count)
      defer { buffer.deallocate() }
      _ = withCStringPointerBuffer(tokens) { tokensPtr in
        needle_sp_tokens_to_ids(
          self.tokenizer,
          UnsafeMutablePointer(mutating: tokensPtr.baseAddress),
          buffer,
          tokens.count
        )
      }
      return (0..<tokens.count).map { NeedleToken.ID(buffer[$0]) }
    }

    public func tokens(from tokenIds: [NeedleToken.ID]) -> [String] {
      let buffer = UnsafeMutableBufferPointer<UnsafeMutablePointer<CChar>?>
        .allocate(capacity: tokenIds.count)
      for i in 0..<buffer.count {
        buffer[i] = .allocate(capacity: 64)
      }
      defer {
        for i in 0..<buffer.count {
          buffer[i]?.deallocate()
        }
        buffer.deallocate()
      }

      _ = tokenIds.map { Int32($0) }
        .withUnsafeBufferPointer { tokenIdsPtr in
          needle_sp_ids_to_tokens(
            self.tokenizer,
            tokenIdsPtr.baseAddress,
            buffer.baseAddress,
            tokenIds.count
          )
        }
      return Array(buffer).map { String(cString: $0!) }
    }

    public func encode(text: String) -> [NeedleToken.ID] {
      var size = 0
      let tokenIds = needle_sp_encode(self.tokenizer, text, &size)
      defer { tokenIds?.deallocate() }
      return Array(UnsafeBufferPointer(start: tokenIds, count: Int(size)))
        .map { NeedleToken.ID($0) }
    }

    public func decode(tokenIds: some Sequence<NeedleToken.ID>) -> String {
      let tokenIds = Array(tokenIds.map { Int32($0) })
      let str = tokenIds.withUnsafeBufferPointer { tokenIdsPtr in
        needle_sp_decode(self.tokenizer, tokenIdsPtr.baseAddress, tokenIdsPtr.count, nil)
      }
      defer { str?.deallocate() }
      return str.map { String(cString: $0) } ?? ""
    }
  }

  // MARK: - NeedleSentencepieceTokenizerError

  public struct NeedleSentencepieceTokenizerError: Hashable, Sendable, Error {
    public let message: String

    fileprivate init() {
      self.message = String(cString: needle_last_error_message())
    }
  }
#endif
