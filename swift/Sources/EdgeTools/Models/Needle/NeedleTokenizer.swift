#if XGrammar
  import EdgeToolsXGrammar
#endif

import HeapModule

#if System
  import SystemPackage
#endif

#if Foundation
  import Foundation
#endif

// MARK: - NeedleSPTokenizer

public struct NeedleSPTokenizer: Sendable {
  public var vocabularySize: Int { self.pieces.count }
  public let bosTokenId: EdgeToolsToken.ID?
  public let eosTokenId: EdgeToolsToken.ID?
  public let unknownTokenId: EdgeToolsToken.ID?
  public let padTokenId: EdgeToolsToken.ID?

  private let pieces: [SentencePiece]
  private let pieceIds: [[UInt8]: EdgeToolsToken.ID]
  private let pieceScores: [[UInt8]: Float]
  private let userDefinedPieces: [[UInt8]]
  private let byteIds: [EdgeToolsToken.ID]?
  private let unknownSurface: String
  private let normalization: Normalization

  public init<Bytes: Collection>(data: Bytes) throws where Bytes.Element == UInt8 {
    do {
      let model = ProtobufMessage(bytes: Array(data))
      let pieces = try model.messages(field: 1).map(SentencePiece.init)
      let trainer = try model.lastMessage(field: 2) ?? ProtobufMessage(bytes: [])
      let normalizer = try model.lastMessage(field: 3) ?? ProtobufMessage(bytes: [])
      let modelType = (try trainer.lastInt32(field: 3)) ?? 1
      let normalizerName = (try normalizer.lastString(field: 1)) ?? ""

      guard modelType == 2 else {
        throw NeedleSPTokenizerError(
          code: .unsupportedModelType,
          message: "NeedleSPTokenizer only supports SentencePiece BPE models."
        )
      }
      guard
        (try normalizer.lastBytes(field: 2) ?? []).isEmpty,
        normalizerName.isEmpty || normalizerName == "identity"
      else {
        throw NeedleSPTokenizerError(
          code: .unsupportedNormalization,
          message: "NeedleSPTokenizer only supports identity normalization."
        )
      }
      guard try model.lastBytes(field: 5) == nil else {
        throw NeedleSPTokenizerError(
          code: .unsupportedDenormalizer,
          message: "NeedleSPTokenizer does not support SentencePiece denormalizers."
        )
      }
      guard !pieces.isEmpty else {
        throw NeedleSPTokenizerError(
          code: .emptyModel,
          message: "SentencePiece model contains no pieces."
        )
      }

      let userDefinedPieces =
        pieces
        .filter { $0.type == .userDefined }
        .map { Array($0.text.utf8) }
        .sorted { $0.count > $1.count }
      guard !userDefinedPieces.contains(where: \.isEmpty) else {
        throw NeedleSPTokenizerError(
          code: .emptyUserDefinedPiece,
          message: "SentencePiece model contains an empty user-defined piece."
        )
      }
      let usesByteFallback = (try trainer.lastBool(field: 35)) ?? false

      self.bosTokenId = Self.tokenId((try trainer.lastInt32(field: 41)) ?? 1, in: pieces)
      self.eosTokenId = Self.tokenId((try trainer.lastInt32(field: 42)) ?? 2, in: pieces)
      self.unknownTokenId = Self.tokenId((try trainer.lastInt32(field: 40)) ?? 0, in: pieces)
      self.padTokenId = Self.tokenId((try trainer.lastInt32(field: 43)) ?? -1, in: pieces)
      self.pieces = pieces
      self.pieceIds = Dictionary(
        pieces.enumerated().map { (Array($0.element.text.utf8), $0.offset) },
        uniquingKeysWith: { first, _ in first }
      )
      self.pieceScores = Dictionary(
        pieces.compactMap { piece in
          piece.type == .normal ? (Array(piece.text.utf8), piece.score) : nil
        },
        uniquingKeysWith: { first, _ in first }
      )
      self.userDefinedPieces = userDefinedPieces
      self.byteIds = try Self.byteIds(in: pieces, enabled: usesByteFallback)
      self.unknownSurface = (try trainer.lastString(field: 44)) ?? " ⁇ "
      self.normalization = Normalization(
        addDummyPrefix: (try normalizer.lastBool(field: 3)) ?? true,
        removeExtraWhitespaces: (try normalizer.lastBool(field: 4)) ?? true,
        escapeWhitespaces: (try normalizer.lastBool(field: 5)) ?? true,
        treatWhitespaceAsSuffix: (try trainer.lastBool(field: 24)) ?? false
      )
    }
  }

  public func encode(text: String) -> [EdgeToolsToken.ID] {
    let normalized = self.normalize(text)
    guard !normalized.isEmpty else { return [] }

    var encoder = BPEEncoder(
      symbols: self.makeSymbols(from: normalized),
      pieceScores: self.pieceScores,
      pieceIds: self.pieceIds,
      byteIds: self.byteIds,
      unknownTokenId: self.unknownTokenId
    )
    return encoder.encode()
  }

  public func decode(tokens: [EdgeToolsToken.ID]) -> String {
    guard tokens.allSatisfy({ self.pieces.indices.contains($0) }) else { return "" }

    var decoder = SentencePieceDecoder(
      pieces: self.pieces,
      unknownSurface: self.unknownSurface,
      normalization: self.normalization
    )
    return decoder.decode(tokens: tokens)
  }

  public func convertTokensToIds(_ tokens: [String]) -> [EdgeToolsToken.ID?] {
    tokens.map { self.pieceIds[Array($0.utf8)] }
  }

  public func convertIdsToTokens(_ ids: [EdgeToolsToken.ID]) -> [String?] {
    ids.map { self.pieces.indices.contains($0) ? self.pieces[$0].text : nil }
  }

  private func normalize(_ text: String) -> String {
    guard !text.isEmpty else { return "" }

    let scalars = text.unicodeScalars
    let normalizedScalars: [Unicode.Scalar]
    if self.normalization.removeExtraWhitespaces {
      normalizedScalars =
        scalars.reduce(into: [Unicode.Scalar]()) { result, scalar in
          if scalar.value == 0x20 {
            if !result.isEmpty && result.last?.value != 0x20 {
              result.append(scalar)
            }
          } else {
            result.append(scalar)
          }
        }
        .dropLast { $0.value == 0x20 }
    } else {
      normalizedScalars = Array(scalars)
    }
    guard !normalizedScalars.isEmpty else { return "" }

    var normalized = normalizedScalars.reduce(into: "") { result, scalar in
      if scalar.value == 0x20 && self.normalization.escapeWhitespaces {
        result += sentencePieceSpaceSymbol
      } else {
        result.unicodeScalars.append(scalar)
      }
    }
    if self.normalization.addDummyPrefix {
      let whitespace = self.normalization.escapeWhitespaces ? sentencePieceSpaceSymbol : " "
      if self.normalization.treatWhitespaceAsSuffix {
        normalized += whitespace
      } else {
        normalized = whitespace + normalized
      }
    }
    return normalized
  }

  private func makeSymbols(from normalized: String) -> [Symbol] {
    let bytes = Array(normalized.utf8)
    var symbols = [Symbol]()
    var offset = 0

    while offset < bytes.count {
      let userDefinedPiece = self.userDefinedPieces.first {
        bytes[offset...].starts(with: $0)
      }
      let byteCount =
        userDefinedPiece?.count
        ?? Self.utf8ScalarLength(startingWith: bytes[offset])
      let end = min(offset + byteCount, bytes.count)
      symbols.append(
        Symbol(
          previous: symbols.isEmpty ? -1 : symbols.count - 1,
          next: end == bytes.count ? -1 : symbols.count + 1,
          frozen: userDefinedPiece != nil,
          bytes: Array(bytes[offset..<end])
        )
      )
      offset = end
    }
    return symbols
  }

  private static func tokenId(_ id: Int32, in pieces: [SentencePiece]) -> EdgeToolsToken.ID? {
    pieces.indices.contains(Int(id)) ? Int(id) : nil
  }

  private static func byteIds(
    in pieces: [SentencePiece],
    enabled: Bool
  ) throws -> [EdgeToolsToken.ID]? {
    guard enabled else { return nil }
    var ids = [EdgeToolsToken.ID?](repeating: nil, count: 256)
    for (id, piece) in pieces.enumerated() where piece.type == .byte {
      if let byte = piece.byteValue {
        ids[Int(byte)] = id
      }
    }
    let byteIds = ids.compactMap { $0 }
    guard byteIds.count == ids.count else {
      throw NeedleSPTokenizerError(
        code: .incompleteByteFallbackVocabulary,
        message: "SentencePiece byte fallback requires all 256 byte pieces."
      )
    }
    return byteIds
  }

  private static func utf8ScalarLength(startingWith byte: UInt8) -> Int {
    switch byte {
    case 0x00...0x7F: 1
    case 0xC0...0xDF: 2
    case 0xE0...0xEF: 3
    default: 4
    }
  }

}

extension NeedleSPTokenizer: EdgeToolsTokenizer {}

#if XGrammar
  extension NeedleSPTokenizer: EdgeToolsXGRTokenizer {
    public func tokenizerInfo(modelVocabularySize: Int? = nil) throws -> XGRTokenizerInfo {
      let vocabularySize = modelVocabularySize ?? self.vocabularySize
      return try XGRTokenizerInfo.needle(tokenizer: self, vocabularySize: vocabularySize)
    }
  }
#endif

#if System
  extension NeedleSPTokenizer {
    public init(modelPath: FilePath) throws {
      do {
        try self.init(data: readFile(at: modelPath))
      } catch let error as NeedleSPTokenizerError {
        throw error
      } catch {
        throw NeedleSPTokenizerError.missingModelFile(at: modelPath.string)
      }
    }
  }
#endif

#if Foundation
  extension NeedleSPTokenizer {
    public init(modelURL: URL) throws {
      guard modelURL.isFileURL, !modelURL.hasDirectoryPath else {
        throw NeedleSPTokenizerError.missingModelFile(at: modelURL.path())
      }
      do {
        try self.init(data: Data(contentsOf: modelURL))
      } catch let error as NeedleSPTokenizerError {
        throw error
      } catch {
        throw NeedleSPTokenizerError.missingModelFile(at: modelURL.path())
      }
    }
  }
#endif

// MARK: - NeedleSPTokenizerError

public struct NeedleSPTokenizerError: Hashable, Sendable, Error {
  public struct Code: RawRepresentable, Hashable, Sendable {
    public let rawValue: String

    public init(rawValue: String) {
      self.rawValue = rawValue
    }

    public static let unsupportedModelType = Self(rawValue: "unsupported-model-type")
    public static let unsupportedNormalization = Self(rawValue: "unsupported-normalization")
    public static let unsupportedDenormalizer = Self(rawValue: "unsupported-denormalizer")
    public static let emptyModel = Self(rawValue: "empty-model")
    public static let emptyUserDefinedPiece = Self(rawValue: "empty-user-defined-piece")
    public static let incompleteByteFallbackVocabulary = Self(
      rawValue: "incomplete-byte-fallback-vocabulary"
    )
    public static let invalidProtobuf = Self(rawValue: "invalid-protobuf")
    public static let missingModelFile = Self(rawValue: "missing-model-file")
    public static let unknownPieceType = Self(rawValue: "unknown-piece-type")
  }

  public let code: Code
  public let message: String

  public init(code: Code, message: String) {
    self.code = code
    self.message = message
  }

  static func invalidProtobuf(message: String) -> Self {
    Self(code: .invalidProtobuf, message: "SentencePiece protobuf \(message)")
  }

  static func missingModelFile(at path: String) -> Self {
    Self(code: .missingModelFile, message: "\(path): file not found")
  }
}

// MARK: - Helpers

private let sentencePieceSpaceSymbol = "▁"

private struct Normalization: Hashable, Sendable {
  let addDummyPrefix: Bool
  let removeExtraWhitespaces: Bool
  let escapeWhitespaces: Bool
  let treatWhitespaceAsSuffix: Bool
}

private struct Symbol {
  var previous: Int
  var next: Int
  let frozen: Bool
  var bytes: [UInt8]
}

private struct MergeCandidate: Comparable, Hashable, Sendable {
  let left: Int
  let right: Int
  let score: Float
  let byteCount: Int

  static func < (lhs: Self, rhs: Self) -> Bool {
    lhs.score < rhs.score || (lhs.score == rhs.score && lhs.left > rhs.left)
  }
}

private struct BPEEncoder {
  var symbols: [Symbol]
  let pieceScores: [[UInt8]: Float]
  let pieceIds: [[UInt8]: EdgeToolsToken.ID]
  let byteIds: [EdgeToolsToken.ID]?
  let unknownTokenId: EdgeToolsToken.ID?
  var merges = Heap<MergeCandidate>()

  mutating func encode() -> [EdgeToolsToken.ID] {
    for right in self.symbols.indices.dropFirst() {
      self.insertCandidate(left: right - 1, right: right)
    }
    while let merge = self.merges.popMax() {
      guard self.canApply(merge) else { continue }
      self.apply(merge)
    }
    return self.tokenIds()
  }

  private func candidate(left: Int, right: Int) -> MergeCandidate? {
    guard
      self.symbols.indices.contains(left),
      self.symbols.indices.contains(right),
      !self.symbols[left].frozen,
      !self.symbols[right].frozen
    else { return nil }

    let bytes = self.symbols[left].bytes + self.symbols[right].bytes
    return self.pieceScores[bytes]
      .map {
        MergeCandidate(
          left: left,
          right: right,
          score: $0,
          byteCount: bytes.count
        )
      }
  }

  private mutating func insertCandidate(left: Int, right: Int) {
    self.candidate(left: left, right: right).map { self.merges.insert($0) }
  }

  private func canApply(_ merge: MergeCandidate) -> Bool {
    !self.symbols[merge.left].bytes.isEmpty
      && !self.symbols[merge.right].bytes.isEmpty
      && self.symbols[merge.left].next == merge.right
      && self.symbols[merge.left].bytes.count + self.symbols[merge.right].bytes.count
        == merge.byteCount
  }

  private mutating func apply(_ merge: MergeCandidate) {
    self.symbols[merge.left].bytes += self.symbols[merge.right].bytes
    self.symbols[merge.left].next = self.symbols[merge.right].next
    if self.symbols[merge.right].next >= 0 {
      self.symbols[self.symbols[merge.right].next].previous = merge.left
    }
    self.symbols[merge.right].bytes.removeAll(keepingCapacity: false)

    self.insertCandidate(left: self.symbols[merge.left].previous, right: merge.left)
    self.insertCandidate(left: merge.left, right: self.symbols[merge.left].next)
  }

  private func tokenIds() -> [EdgeToolsToken.ID] {
    var ids = [EdgeToolsToken.ID]()
    var index = 0
    while index >= 0 {
      let symbol = self.symbols[index]
      if let id = self.pieceIds[symbol.bytes] {
        ids.append(id)
      } else if let byteIds = self.byteIds {
        ids.append(contentsOf: symbol.bytes.map { byteIds[Int($0)] })
      } else if let unknownTokenId = self.unknownTokenId {
        ids.append(unknownTokenId)
      }
      index = symbol.next
    }
    return ids
  }
}

private struct SentencePieceDecoder {
  private let pieces: [SentencePiece]
  private let unknownSurface: String
  private let normalization: Normalization
  private var decoded = [UInt8]()
  private var byteRun = [UInt8]()
  private var expectsBeginningWhitespace = true

  init(
    pieces: [SentencePiece],
    unknownSurface: String,
    normalization: Normalization
  ) {
    self.pieces = pieces
    self.unknownSurface = unknownSurface
    self.normalization = normalization
  }

  mutating func decode(tokens: [EdgeToolsToken.ID]) -> String {
    for token in tokens {
      self.append(self.pieces[token])
    }
    self.appendByteRun()
    return String(decoding: self.decoded, as: UTF8.self)
  }

  private mutating func append(_ piece: SentencePiece) {
    if piece.type == .byte {
      if let byte = piece.byteValue {
        self.byteRun.append(byte)
      }
      return
    }

    self.appendByteRun()
    if !self.decoded.isEmpty {
      self.expectsBeginningWhitespace = false
    }

    switch piece.type {
    case .unknown:
      self.decoded.append(contentsOf: self.unknownSurface.utf8)
    case .normal, .userDefined:
      self.appendText(piece.text)
    case .control, .unused, .byte:
      break
    }
  }

  private mutating func appendText(_ piece: String) {
    var text = piece
    if self.expectsBeginningWhitespace
      && (self.normalization.addDummyPrefix || self.normalization.removeExtraWhitespaces)
      && text.hasPrefix(sentencePieceSpaceSymbol)
    {
      text.removeFirst()
      if !self.normalization.removeExtraWhitespaces {
        self.expectsBeginningWhitespace = false
      }
    }
    self.decoded.append(
      contentsOf: text.replacing(sentencePieceSpaceSymbol, with: " ").utf8
    )
  }

  private mutating func appendByteRun() {
    guard !self.byteRun.isEmpty else { return }
    self.decoded.append(contentsOf: String(decoding: self.byteRun, as: UTF8.self).utf8)
    self.byteRun.removeAll(keepingCapacity: true)
  }
}

private struct SentencePiece: Hashable, Sendable {
  enum PieceType: Int32, Hashable, Sendable {
    case normal = 1
    case unknown = 2
    case control = 3
    case userDefined = 4
    case unused = 5
    case byte = 6
  }

  var text = ""
  var score: Float = 0
  var type = PieceType.normal

  var byteValue: UInt8? {
    let bytes = Array(self.text.utf8)
    guard
      bytes.count == 6,
      bytes.starts(with: Array("<0x".utf8)),
      bytes[5] == 0x3E
    else { return nil }
    return UInt8(String(decoding: bytes[3...4], as: UTF8.self), radix: 16)
  }

  init(message: ProtobufMessage) throws {
    self.text = (try message.lastString(field: 1)) ?? ""
    self.score = (try message.lastFixed32(field: 2)).map { Float(bitPattern: $0) } ?? 0
    let rawType = (try message.lastInt32(field: 3)) ?? 1
    guard let type = PieceType(rawValue: rawType) else {
      throw NeedleSPTokenizerError(
        code: .unknownPieceType,
        message: "SentencePiece protobuf contains an unknown piece type."
      )
    }
    self.type = type
  }
}

// MARK: - Needle Tokenizer Integration

extension EdgeToolsTokenizer {
  public var needleToolsToken: String { "<tools>" }

  public var needleToolsTokenId: EdgeToolsToken.ID? {
    self.convertTokenToId(self.needleToolsToken)
  }

  public var needleToolCallToken: String { "<tool_call>" }

  public var needleToolCallTokenId: EdgeToolsToken.ID? {
    self.convertTokenToId(self.needleToolCallToken)
  }
}
