#if SwiftNeedleSentencepiece
  import Foundation
  import CNeedleSentencepiece
  import Tokenizers
  import Hub

  // MARK: - NeedleSentencepieceTokenizer

  public final class NeedleSPTokenizingModel: TokenizingModel {
    private static let tokenBufferSize = 256

    public let tokenizer: needle_sp_tokenizer_t

    public var vocabSize: Int {
      needle_sp_tokenizer_vocab_size(self.tokenizer)
    }

    public init(modelURL: URL) throws {
      let nativePath = modelURL.withUnsafeFileSystemRepresentation { $0 }
      guard let tokenizer = needle_sp_tokenizer_init_from_file(nativePath) else {
        throw NeedleSentencepieceTokenizerError()
      }
      self.tokenizer = tokenizer
    }

    public init(tokenizer: consuming needle_sp_tokenizer_t) {
      self.tokenizer = tokenizer
    }

    deinit { needle_sp_tokenizer_destroy(self.tokenizer) }

    public func encodedVocab() -> [String] {
      self.withCCharStringBuffer(count: self.vocabSize) { buffer in
        _ = needle_sp_tokenizer_encoded_vocab(self.tokenizer, buffer.baseAddress)
        return Array(buffer).map { String(cString: $0!) }
      }
    }

    public func tokenIds(from tokens: [String]) -> [NeedleToken.ID] {
      let buffer = UnsafeMutablePointer<Int32>.allocate(capacity: tokens.count)
      defer { buffer.deallocate() }
      _ = withCStringPointerBuffer(tokens) { tokensPtr in
        needle_sp_tokenizer_tokens_to_ids(
          self.tokenizer,
          UnsafeMutablePointer(mutating: tokensPtr.baseAddress),
          buffer,
          tokens.count
        )
      }
      return (0..<tokens.count).map { NeedleToken.ID(buffer[$0]) }
    }

    public func tokens(from tokenIds: [NeedleToken.ID]) -> [String] {
      self.withCCharStringBuffer(count: tokenIds.count) { buffer in
        tokenIds.map { Int32($0) }
          .withUnsafeBufferPointer { tokenIdsPtr in
            _ = needle_sp_tokenizer_ids_to_tokens(
              self.tokenizer,
              tokenIdsPtr.baseAddress,
              buffer.baseAddress,
              buffer.count
            )
          }
        return Array(buffer).map { String(cString: $0!) }
      }
    }

    public func encode(text: String) -> [NeedleToken.ID] {
      var size = 0
      let tokenIds = needle_sp_tokenizer_encode(self.tokenizer, text, &size)
      defer { tokenIds?.deallocate() }
      return Array(UnsafeBufferPointer(start: tokenIds, count: Int(size)))
        .map { NeedleToken.ID($0) }
    }

    public func decode(tokenIds: some Sequence<NeedleToken.ID>) -> String {
      let tokenIds = Array(tokenIds.map { Int32($0) })
      let str = tokenIds.withUnsafeBufferPointer { tokenIdsPtr in
        needle_sp_tokenizer_decode(self.tokenizer, tokenIdsPtr.baseAddress, tokenIdsPtr.count, nil)
      }
      defer { str?.deallocate() }
      return str.map { String(cString: $0) } ?? ""
    }

    public func tokenize(text: String) -> [String] {
      self.tokens(from: self.encode(text: text))
    }

    public func convertTokenToId(_ token: String) -> Int? {
      guard token != self.unknownToken else { return self.unknownTokenId }
      guard let id = self.tokenIds(from: [token]).first.map({ Int($0) }) else { return nil }
      return id == self.unknownTokenId ? nil : id
    }

    public func convertIdToToken(_ id: Int) -> String? {
      guard id != self.unknownTokenId else { return nil }
      guard let token = self.tokens(from: [id]).first, !token.isEmpty else { return nil }
      return token
    }

    public var bosToken: String? {
      self.bosTokenId.flatMap(self.convertIdToToken)
    }

    public var bosTokenId: NeedleToken.ID? {
      Int(needle_sp_tokenizer_bos_token_id(self.tokenizer))
    }

    public var eosToken: String? {
      self.eosTokenId.flatMap(self.convertIdToToken)
    }

    public var eosTokenId: NeedleToken.ID? {
      Int(needle_sp_tokenizer_eos_token_id(self.tokenizer))
    }

    public var unknownToken: String? {
      self.unknownTokenId.flatMap { self.tokens(from: [$0]).first }
    }

    public var unknownTokenId: Int? {
      Int(needle_sp_tokenizer_unk_token_id(self.tokenizer))
    }

    public var padTokenId: NeedleToken.ID {
      Int(needle_sp_tokenizer_pad_token_id(self.tokenizer))
    }

    public var fuseUnknownTokens: Bool { false }

    private func withCCharStringBuffer<Result>(
      count: Int,
      _ body: (UnsafeMutableBufferPointer<UnsafeMutablePointer<CChar>?>) throws -> Result
    ) rethrows -> Result {
      let buffer = UnsafeMutableBufferPointer<UnsafeMutablePointer<CChar>?>
        .allocate(capacity: count * MemoryLayout<CChar>.size)
      for i in 0..<buffer.count {
        buffer[i] = .allocate(capacity: Self.tokenBufferSize * MemoryLayout<CChar>.size)
      }
      defer {
        for i in 0..<buffer.count {
          buffer[i]?.deallocate()
        }
        buffer.deallocate()
      }
      return try body(buffer)
    }
  }

  // MARK: - NeedleSentencepieceTokenizerError

  public struct NeedleSentencepieceTokenizerError: Hashable, Sendable, Error {
    public let message: String

    fileprivate init() {
      self.message = String(cString: needle_sp_last_error_message())
    }
  }
#endif
