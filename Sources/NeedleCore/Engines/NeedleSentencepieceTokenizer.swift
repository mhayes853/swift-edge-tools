#if SwiftNeedleSentencepiece
  import Foundation
  private import Sentencepiece

  // MARK: - NeedleSentencepieceTokenizer

  public struct NeedleSentencepieceTokenizer: ~Copyable {
    private let tokenizer: UnsafeMutableRawPointer

    public var unkTokenId: NeedleToken.ID { Int(spm_unk_id(self.tokenizer)) }
    public var bosTokenId: NeedleToken.ID { Int(spm_bos_id(self.tokenizer)) }
    public var eosTokenId: NeedleToken.ID { Int(spm_eos_id(self.tokenizer)) }
    public var padTokenId: NeedleToken.ID { Int(spm_pad_id(self.tokenizer)) }

    public init(modelURL: URL) throws {
      self.tokenizer = spm_new_sentencepiece_processor()
      let nativePath = modelURL.withUnsafeFileSystemRepresentation { $0 }
      if !spm_load_model(self.tokenizer, nativePath) {
        throw NeedleSentencepieceTokenizerError.failedToLoad
      }
    }

    deinit { self.tokenizer.deallocate() }

    public func encode(text: String) -> [NeedleToken.ID] {
      var size = Int32(0)
      let ptr = spm_encode(self.tokenizer, text, &size)
      defer { ptr?.deallocate() }
      return Array(UnsafeBufferPointer(start: ptr, count: Int(size))).map { NeedleToken.ID($0) }
    }

    public func decode(tokenIds: some Sequence<NeedleToken.ID>) -> String {
      let tokenIds = Array(tokenIds.map { Int32($0) })
      let str = tokenIds.withUnsafeBufferPointer { tokenIdsPtr in
        spm_decode(self.tokenizer, tokenIdsPtr.baseAddress, Int32(tokenIdsPtr.count))
      }
      defer { str?.deallocate() }
      return str.map { String(cString: $0) } ?? ""
    }
  }

  // MARK: - NeedleSentencepieceTokenizerError

  public enum NeedleSentencepieceTokenizerError: Hashable, Sendable, Error {
    case failedToLoad
  }
#endif
