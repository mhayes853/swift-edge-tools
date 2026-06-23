#if SwiftNeedleSentencepiece
  import Foundation
  import CNeedleSentencepiece
  import Tokenizers

  // MARK: - NeedleSPTokenizer

  public final class NeedleSPTokenizer: Tokenizer {
    private static let tokenBufferSize = 256

    private let tokenizer: RecursiveLock<needle_sp_tokenizer_t>

    public var hasChatTemplate: Bool { false }

    public var bosToken: String? {
      self.bosTokenId.flatMap(self.convertIdToToken)
    }

    public var bosTokenId: NeedleToken.ID? {
      self.withRawTokenizer { tokenizer in
        Int(needle_sp_tokenizer_bos_token_id(tokenizer))
      }
    }

    public var eosToken: String? {
      self.eosTokenId.flatMap(self.convertIdToToken)
    }

    public var eosTokenId: NeedleToken.ID? {
      self.withRawTokenizer { tokenizer in
        Int(needle_sp_tokenizer_eos_token_id(tokenizer))
      }
    }

    public var unknownToken: String? {
      self.unknownTokenId.flatMap(self.convertIdToToken)
    }

    public var unknownTokenId: Int? {
      self.withRawTokenizer { tokenizer in
        Int(needle_sp_tokenizer_unk_token_id(tokenizer))
      }
    }

    public var toolsToken: String {
      "<tools>"
    }

    public var toolsTokenId: Int? {
      self.convertTokenToId(self.toolsToken)
    }

    public var toolCallToken: String {
      "<tool_call>"
    }

    public var toolCallTokenId: Int? {
      self.convertTokenToId(self.toolCallToken)
    }

    public var fuseUnknownTokens: Bool { false }

    public init(modelURL: URL) throws {
      let nativePath = modelURL.withUnsafeFileSystemRepresentation { String(cString: $0!) }
      guard let tokenizer = needle_sp_tokenizer_init_from_file(nativePath) else {
        throw NeedleSPTokenizerError()
      }
      self.tokenizer = RecursiveLock(tokenizer)
    }

    public init(tokenizer: consuming sending needle_sp_tokenizer_t) {
      self.tokenizer = RecursiveLock(tokenizer)
    }

    deinit {
      self.tokenizer.withLock { needle_sp_tokenizer_destroy($0) }
    }

    public func withRawTokenizer<T, E: Error>(
      _ body: (needle_sp_tokenizer_t) throws(E) -> sending T
    ) throws(E) -> sending T {
      try self.tokenizer.withLock { tokenizer throws(E) in try body(tokenizer) }
    }

    public func encode(text: String) -> [Int] {
      self.encode(text: text, addSpecialTokens: true)
    }

    public func tokenize(text: String) -> [String] {
      self.tokens(from: self.encode(text: text, addSpecialTokens: false))
    }

    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
      self.withRawTokenizer { tokenizer in
        var size = 0
        let tokenIds = needle_sp_tokenizer_encode(tokenizer, text, &size)
        defer { tokenIds?.deallocate() }
        return Array(UnsafeBufferPointer(start: tokenIds, count: Int(size))).map(Int.init)
      }
    }

    public func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
      let filteredTokenIds = skipSpecialTokens ? self.skippingSpecialTokens(in: tokens) : tokens
      let tokenIds = filteredTokenIds.map(Int32.init)
      let string = self.tokenizer.withLock { tokenizer in
        tokenIds.withUnsafeBufferPointer { tokenIdsPtr in
          needle_sp_tokenizer_decode(tokenizer, tokenIdsPtr.baseAddress, tokenIdsPtr.count, nil)
        }
      }
      defer { string?.deallocate() }
      return string.map { String(cString: $0) } ?? ""
    }

    public func convertTokenToId(_ token: String) -> Int? {
      self.convertTokensToIds([token])[0]
    }

    public func convertTokensToIds(_ tokens: [String]) -> [Int?] {
      zip(tokens, self.tokenIds(from: tokens))
        .map { token, id in
          guard token != self.unknownToken else { return id }
          return id == self.unknownTokenId ? nil : id
        }
    }

    public func convertIdToToken(_ id: Int) -> String? {
      self.convertIdsToTokens([id])[0]
    }

    public func convertIdsToTokens(_ ids: [Int]) -> [String?] {
      self.tokens(from: ids).map { $0.isEmpty ? nil : $0 }
    }

    public func applyChatTemplate(messages: [Message]) throws -> [Int] {
      throw TokenizerError.missingChatTemplate
    }

    public func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] {
      throw TokenizerError.missingChatTemplate
    }

    public func applyChatTemplate(
      messages: [Message],
      tools: [ToolSpec]?,
      additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
      throw TokenizerError.missingChatTemplate
    }

    public func applyChatTemplate(
      messages: [Message],
      chatTemplate: ChatTemplateArgument
    ) throws -> [Int] {
      throw TokenizerError.missingChatTemplate
    }

    public func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
      throw TokenizerError.missingChatTemplate
    }

    public func applyChatTemplate(
      messages: [Message],
      chatTemplate: ChatTemplateArgument?,
      addGenerationPrompt: Bool,
      truncation: Bool,
      maxLength: Int?,
      tools: [ToolSpec]?
    ) throws -> [Int] {
      throw TokenizerError.missingChatTemplate
    }

    public func applyChatTemplate(
      messages: [Message],
      chatTemplate: ChatTemplateArgument?,
      addGenerationPrompt: Bool,
      truncation: Bool,
      maxLength: Int?,
      tools: [ToolSpec]?,
      additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
      throw TokenizerError.missingChatTemplate
    }

    private var padTokenId: NeedleToken.ID {
      self.withRawTokenizer { tokenizer in
        Int(needle_sp_tokenizer_pad_token_id(tokenizer))
      }
    }

    private func tokenIds(from tokens: [String]) -> [Int] {
      let buffer = UnsafeMutablePointer<Int32>.allocate(capacity: tokens.count)
      defer { buffer.deallocate() }
      _ = withCStringPointerBuffer(tokens) { tokensPtr in
        self.withRawTokenizer { tokenizer in
          needle_sp_tokenizer_tokens_to_ids(
            tokenizer,
            UnsafeMutablePointer(mutating: tokensPtr.baseAddress),
            buffer,
            tokens.count
          )
        }
      }
      return (0..<tokens.count).map { Int(buffer[$0]) }
    }

    private func tokens(from tokenIds: [Int]) -> [String] {
      self.withCCharStringBuffer(count: tokenIds.count) { buffer in
        tokenIds.map(Int32.init)
          .withUnsafeBufferPointer { tokenIdsPtr in
            self.withRawTokenizer { tokenizer in
              _ = needle_sp_tokenizer_ids_to_tokens(
                tokenizer,
                tokenIdsPtr.baseAddress,
                buffer.baseAddress,
                buffer.count
              )
            }
          }
        return Array(buffer.prefix(tokenIds.count)).map { String(cString: $0!) }
      }
    }

    private func skippingSpecialTokens(in tokenIds: [Int]) -> [Int] {
      let specialTokenIds = [
        self.bosTokenId,
        self.eosTokenId,
        self.unknownTokenId,
        self.padTokenId,
        self.toolsTokenId,
        self.toolCallTokenId
      ]
      .compactMap { $0 }
      guard !specialTokenIds.isEmpty else { return tokenIds }
      return tokenIds.filter { !specialTokenIds.contains($0) }
    }

    private static func sendingTokenizer(
      _ tokenizer: consuming needle_sp_tokenizer_t
    ) -> sending needle_sp_tokenizer_t {
      UnsafeMutableRawPointer(bitPattern: Int(bitPattern: tokenizer))!
    }

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

  public struct NeedleSPTokenizerError: Hashable, Sendable, Error {
    public let message: String

    fileprivate init() {
      self.message = String(cString: needle_sp_last_error_message())
    }
  }
#endif
