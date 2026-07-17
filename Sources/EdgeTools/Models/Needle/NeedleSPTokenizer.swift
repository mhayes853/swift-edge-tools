import HeapModule

// MARK: - NeedleSPTokenizer

public struct NeedleSPTokenizer: Sendable {
  public var vocabularySize: Int { self.pieces.count }
  public let bosTokenId: EdgeToolsToken.ID?
  public let eosTokenId: EdgeToolsToken.ID?
  public let unknownTokenId: EdgeToolsToken.ID?
  public let padTokenId: EdgeToolsToken.ID?

  private let pieces: [SentencePiece]
  private let pieceIds: [PieceBytes: EdgeToolsToken.ID]
  private let mergePieces: [PieceBytes: MergePiece]
  private let userDefinedPieces: [PieceBytes]
  private let byteIds: [EdgeToolsToken.ID]?
  private let unknownSurface: String
  private let normalization: Normalization

  public init<Bytes: Collection>(data: Bytes) throws where Bytes.Element == UInt8 {
    let model = try SentencePieceModel(bytes: Array(data))

    let pieceIds = Dictionary(
      uniqueKeysWithValues: model.pieces.enumerated()
        .map { (PieceBytes($0.element.text.utf8), $0.offset) }
    )
    let mergePieces = Dictionary(
      uniqueKeysWithValues: model.pieces.compactMap { piece in
        piece.type == .normal ? (PieceBytes(piece.text.utf8), piece.score) : nil
      }
    )
    let userDefinedPieces = model.pieces
      .filter { $0.type == .userDefined }
      .map { PieceBytes($0.text.utf8) }
      .sorted { $0.count > $1.count }
    let byteIds = try model.byteIds()

    self.bosTokenId = Self.optionalTokenId((try model.trainer.lastInt32(field: 41)) ?? 1)
    self.eosTokenId = Self.optionalTokenId((try model.trainer.lastInt32(field: 42)) ?? 2)
    self.unknownTokenId = Self.optionalTokenId((try model.trainer.lastInt32(field: 40)) ?? 0)
    self.padTokenId = Self.optionalTokenId((try model.trainer.lastInt32(field: 43)) ?? -1)
    self.pieces = model.pieces
    self.pieceIds = pieceIds
    self.mergePieces = mergePieces
    self.userDefinedPieces = userDefinedPieces
    self.byteIds = byteIds
    self.unknownSurface = (try model.trainer.lastString(field: 44)) ?? " ⁇ "
    self.normalization = Normalization(
      addDummyPrefix: (try model.normalizer.lastBool(field: 3)) ?? true,
      removeExtraWhitespaces: (try model.normalizer.lastBool(field: 4)) ?? true,
      escapeWhitespaces: (try model.normalizer.lastBool(field: 5)) ?? true,
      treatWhitespaceAsSuffix: (try model.trainer.lastBool(field: 24)) ?? false
    )
  }

  public func encode(text: String) -> [EdgeToolsToken.ID] {
    let normalized = self.normalize(text)
    guard !normalized.isEmpty else { return [] }

    var encoder = BPEEncoder(
      symbols: self.makeSymbols(from: normalized),
      mergePieces: self.mergePieces,
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
    tokens.map { self.pieceIds[PieceBytes($0.utf8)] }
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

  private static func optionalTokenId(_ id: Int32) -> EdgeToolsToken.ID? {
    id >= 0 ? Int(id) : nil
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

// MARK: - NeedleSPTokenizerError

public struct NeedleSPTokenizerError: Hashable, Sendable, Error {
  public let message: String

  public static func unsupportedModelType() -> Self {
    Self(message: "NeedleSPTokenizer only supports SentencePiece BPE models.")
  }

  public static func unsupportedNormalizer() -> Self {
    Self(message: "NeedleSPTokenizer only supports identity normalization.")
  }

  public static func unsupportedDenormalizer() -> Self {
    Self(message: "NeedleSPTokenizer does not support SentencePiece denormalizers.")
  }

  public static func emptyPiece() -> Self {
    Self(message: "SentencePiece model contains an empty user-defined piece.")
  }

  public static func duplicatePieces() -> Self {
    Self(message: "SentencePiece model contains duplicate pieces.")
  }

  public static func invalidUnknownPiece() -> Self {
    Self(message: "SentencePiece model must contain a valid unknown piece at its configured ID.")
  }

  public static func incompleteByteFallbackVocabulary() -> Self {
    Self(message: "SentencePiece byte fallback requires all 256 byte pieces.")
  }

  public static func malformedProtobuf(_ reason: String) -> Self {
    Self(message: "SentencePiece protobuf \(reason)")
  }

  public static func fileNotFound(atPath path: String) -> Self {
    Self(message: "\(path): file not found")
  }

  private init(message: String) {
    self.message = message
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

private typealias PieceBytes = [UInt8]

private typealias MergePiece = Float

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
  private var symbols: [Symbol]
  private let mergePieces: [PieceBytes: MergePiece]
  private let pieceIds: [PieceBytes: EdgeToolsToken.ID]
  private let byteIds: [EdgeToolsToken.ID]?
  private let unknownTokenId: EdgeToolsToken.ID?
  private var merges = Heap<MergeCandidate>()

  init(
    symbols: [Symbol],
    mergePieces: [PieceBytes: MergePiece],
    pieceIds: [PieceBytes: EdgeToolsToken.ID],
    byteIds: [EdgeToolsToken.ID]?,
    unknownTokenId: EdgeToolsToken.ID?
  ) {
    self.symbols = symbols
    self.mergePieces = mergePieces
    self.pieceIds = pieceIds
    self.byteIds = byteIds
    self.unknownTokenId = unknownTokenId
  }

  mutating func encode() -> [EdgeToolsToken.ID] {
    self.seedMerges()
    self.mergeSymbols()
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
    return self.mergePieces[PieceBytes(bytes)]
      .map {
        MergeCandidate(
          left: left,
          right: right,
          score: $0,
          byteCount: bytes.count
        )
      }
  }

  private mutating func seedMerges() {
    guard self.symbols.count > 1 else { return }
    for right in 1..<self.symbols.count {
      if let candidate = self.candidate(left: right - 1, right: right) {
        self.merges.insert(candidate)
      }
    }
  }

  private mutating func mergeSymbols() {
    while let merge = self.merges.popMax() {
      guard self.canApply(merge) else { continue }
      self.apply(merge)
    }
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

    if let previous = self.candidate(
      left: self.symbols[merge.left].previous,
      right: merge.left
    ) {
      self.merges.insert(previous)
    }
    if let next = self.candidate(left: merge.left, right: self.symbols[merge.left].next) {
      self.merges.insert(next)
    }
  }

  private func tokenIds() -> [EdgeToolsToken.ID] {
    var ids = [EdgeToolsToken.ID]()
    var index = 0
    while index >= 0 {
      let symbol = self.symbols[index]
      if let id = self.pieceIds[PieceBytes(symbol.bytes)] {
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

private struct SentencePieceModel {
  let pieces: [SentencePiece]
  let trainer: ProtobufMessage
  let normalizer: ProtobufMessage

  init(bytes: [UInt8]) throws {
    do {
      let message = ProtobufMessage(bytes: bytes)
      self.pieces = try message.messages(field: 1).map(SentencePiece.init)
      self.trainer = try message.lastMessage(field: 2) ?? ProtobufMessage(bytes: [])
      self.normalizer = try message.lastMessage(field: 3) ?? ProtobufMessage(bytes: [])
      let hasDenormalizer = try message.lastBytes(field: 5) != nil
      try self.validate(hasDenormalizer: hasDenormalizer)
    } catch let error as ProtobufReaderError {
      throw NeedleSPTokenizerError.malformedProtobuf(error.message)
    }
  }

  private func validate(hasDenormalizer: Bool) throws {
    let modelType = (try self.trainer.lastInt32(field: 3)) ?? 1
    let normalizerName = (try self.normalizer.lastString(field: 1)) ?? ""
    let precompiledMap = (try self.normalizer.lastBytes(field: 2)) ?? []
    let unknownId = (try self.trainer.lastInt32(field: 40)) ?? 0
    let byteFallback = (try self.trainer.lastBool(field: 35)) ?? false

    guard modelType == 2 else {
      throw NeedleSPTokenizerError.unsupportedModelType()
    }
    guard precompiledMap.isEmpty, normalizerName.isEmpty || normalizerName == "identity" else {
      throw NeedleSPTokenizerError.unsupportedNormalizer()
    }
    guard !hasDenormalizer else {
      throw NeedleSPTokenizerError.unsupportedDenormalizer()
    }
    guard !self.pieces.contains(where: { $0.type == .userDefined && $0.text.isEmpty }) else {
      throw NeedleSPTokenizerError.emptyPiece()
    }
    guard Set(self.pieces.map { PieceBytes($0.text.utf8) }).count == self.pieces.count else {
      throw NeedleSPTokenizerError.duplicatePieces()
    }
    guard self.pieces.indices.contains(Int(unknownId)), self.pieces[Int(unknownId)].type == .unknown else {
      throw NeedleSPTokenizerError.invalidUnknownPiece()
    }
    if byteFallback, try self.byteIds() == nil {
      throw NeedleSPTokenizerError.incompleteByteFallbackVocabulary()
    }
  }

  func byteIds() throws -> [EdgeToolsToken.ID]? {
    guard (try self.trainer.lastBool(field: 35)) ?? false else { return nil }
    var ids = [EdgeToolsToken.ID?](repeating: nil, count: 256)
    for (id, piece) in self.pieces.enumerated() where piece.type == .byte {
      guard let byte = piece.byteValue else { continue }
      ids[Int(byte)] = id
    }
    let byteIds = ids.compactMap { $0 }
    return byteIds.count == ids.count ? byteIds : nil
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
      throw NeedleSPTokenizerError.malformedProtobuf("contains an unknown piece type.")
    }
    self.type = type
  }
}

// MARK: - ProtobufReaderError

extension ProtobufReaderError {
  fileprivate var message: String {
    switch self {
    case .truncated: "is truncated."
    case .invalidTag: "contains an invalid tag."
    case .overflowingVarint: "contains an overflowing varint."
    case .fieldTooLarge: "contains a field that is too large."
    case .invalidUTF8: "contains invalid UTF-8."
    case .unsupportedWireType(let wire): "contains unsupported wire type \(wire)."
    }
  }
}
