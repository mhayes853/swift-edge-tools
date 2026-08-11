#if XGrammar
  import EdgeToolsXGrammar
#endif

#if Foundation
  import _EdgeToolsFoundation
#endif

// MARK: - NeedleModelConfiguration

public struct NeedleModelConfiguration: Hashable, Sendable {
  public var vocabularySize: Int = .needleVocabularySize
  public var dimensions: Int = 512
  public var hiddenDimensions: Int = 512
  public var attentionHeads: Int = 8
  public var kvHeads: Int = 4
  public var encoderLayers: Int = 12
  public var decoderLayers: Int = 8
  public var hiddenLayers: Int = 8
  public var padTokenId: EdgeToolsToken.ID = 0
  public var decoderStartTokenId: EdgeToolsToken.ID = 1
  public var tieWordEmbeddings: Bool = true
  public var maxSeqLen: Int?
  public var maxPositionEmbeddings: Int?
  private var decoderMaxLengthValue: Int?
  private var dtypeValue: String?
  private var torchDTypeValue: String?
  private var ropeThetaValue: Float?
  private var rmsNormEpsValue: Float?

  public var ropeTheta: Float {
    get { self.ropeThetaValue ?? 10000.0 }
    set { self.ropeThetaValue = newValue }
  }

  public var rmsNormEps: Float {
    get { self.rmsNormEpsValue ?? 1e-6 }
    set { self.rmsNormEpsValue = newValue }
  }

  public var dtype: String {
    get { self.dtypeValue ?? self.torchDTypeValue ?? "bfloat16" }
    set { self.dtypeValue = newValue }
  }

  public var attentionHeadDimensions: Int {
    self.dimensions / self.attentionHeads
  }

  public var kvDimensions: Int {
    self.kvHeads * self.attentionHeadDimensions
  }

  public var encoderMaxLength: Int {
    self.maxSeqLen ?? self.maxPositionEmbeddings ?? 1024
  }

  public var decoderMaxLength: Int {
    get { self.decoderMaxLengthValue ?? min(self.encoderMaxLength, 512) }
    set { self.decoderMaxLengthValue = newValue }
  }

  public init(
    vocabularySize: Int = .needleVocabularySize,
    dimensions: Int = 512,
    hiddenDimensions: Int = 512,
    attentionHeads: Int = 8,
    kvHeads: Int = 4,
    encoderLayers: Int = 12,
    decoderLayers: Int = 8,
    hiddenLayers: Int = 8,
    ropeTheta: Float = 10000.0,
    rmsNormEps: Float = 1e-6,
    padTokenId: EdgeToolsToken.ID = 0,
    decoderStartTokenId: EdgeToolsToken.ID = 1,
    tieWordEmbeddings: Bool = true,
    maxSeqLen: Int? = nil,
    maxPositionEmbeddings: Int? = nil,
    decoderMaxLength: Int? = nil,
    dtype: String = "bfloat16"
  ) {
    self.vocabularySize = vocabularySize
    self.dimensions = dimensions
    self.hiddenDimensions = hiddenDimensions
    self.attentionHeads = attentionHeads
    self.kvHeads = kvHeads
    self.encoderLayers = encoderLayers
    self.decoderLayers = decoderLayers
    self.hiddenLayers = hiddenLayers
    self.ropeTheta = ropeTheta
    self.rmsNormEps = rmsNormEps
    self.padTokenId = padTokenId
    self.decoderStartTokenId = decoderStartTokenId
    self.tieWordEmbeddings = tieWordEmbeddings
    self.maxSeqLen = maxSeqLen
    self.maxPositionEmbeddings = maxPositionEmbeddings
    self.decoderMaxLengthValue = decoderMaxLength
    self.dtypeValue = dtype
    self.torchDTypeValue = nil
  }

}

extension NeedleModelConfiguration: EdgeToolsCodable {
  #if !$Embedded
    private enum CodingKeys: String, CodingKey {
      case vocabularySize = "vocab_size"
      case dimensions = "d_model"
      case hiddenDimensions = "hidden_size"
      case attentionHeads = "num_attention_heads"
      case encoderLayers = "num_encoder_layers"
      case decoderLayers = "num_decoder_layers"
      case hiddenLayers = "num_hidden_layers"
      case kvHeads = "num_kv_heads"
      case ropeThetaValue = "rope_theta"
      case rmsNormEpsValue = "rms_norm_eps"
      case padTokenId = "pad_token_id"
      case decoderStartTokenId = "decoder_start_token_id"
      case tieWordEmbeddings = "tie_word_embeddings"
      case maxSeqLen = "max_seq_len"
      case maxPositionEmbeddings = "max_position_embeddings"
      case decoderMaxLengthValue = "decoder_max_length"
      case dtypeValue = "dtype"
      case torchDTypeValue = "torch_dtype"
    }
  #endif
}

// MARK: - NeedleNumerics

extension Float {
  static let needleClippingMagnitude: Self = 65_500
}

extension Int {
  public static let needleVocabularySize: Self = 8192
}

// MARK: - Loading

#if Foundation
  extension NeedleModelConfiguration {
    static func decode(in directory: URL, decoder: JSONDecoder = JSONDecoder()) throws -> Self? {
      try decodeModelConfiguration(Self.self, in: directory, decoder: decoder)
    }
  }
#endif

// MARK: - NeedlePrompt

public struct NeedlePrompt: Hashable, Sendable {
  public var system: String
  public var user: String

  public init(system: String, user: String) {
    self.system = system
    self.user = user
  }
}

// MARK: - Formatting

extension NeedlePrompt {
  public func formatted(tools: [EdgeToolDefinition]) throws -> String {
    let separator = self.system.isEmpty || self.user.isEmpty ? "" : "\n\n"
    let toolsSchema =
      try tools
      .filter { $0.includesSchemaInInstructions }
      .map { $0.needleNormalized() }
      .needlePromptEncoded()
    return "\(self.system)\(separator)\(self.user)<tools>\(toolsSchema)"
  }

  public func tokenized(
    tools: [EdgeToolDefinition],
    using tokenizer: some EdgeToolsTokenizer
  ) throws -> [EdgeToolsToken] {
    let tokenIds = tokenizer.encode(text: try self.formatted(tools: tools))
    let tokens = tokenizer.convertIdsToTokens(tokenIds)
    return zip(tokenIds, tokens)
      .compactMap { (tokenId, token) in
        token.map { EdgeToolsToken(id: tokenId, stringValue: $0) }
      }
  }
}

// MARK: - EdgeToolDefinition

extension EdgeToolDefinition {
  public func needleNormalized() -> Self {
    var definition = self
    definition.name = self.name.snakeCased()
    return definition
  }

  fileprivate func needlePromptEncoded() throws -> String {
    let name = OrderedKeyJSONWriter.encode(.string(self.name))
    let description = OrderedKeyJSONWriter.encode(.string(self.description))
    let arguments = self.arguments.orderedJSONString()
    return #"{"name":\#(name),"description":\#(description),"arguments":\#(arguments)}"#
  }
}

extension Sequence where Element == EdgeToolDefinition {
  fileprivate func needlePromptEncoded() throws -> String {
    let definitions = try self.map { try $0.needlePromptEncoded() }
    return "[\(definitions.joined(separator: ","))]"
  }
}

// MARK: - NeedleGenerationParser

public struct NeedleGenerationParser: EdgeToolsGenerationParser, Sendable {
  private enum Region {
    case text
    case tool
  }

  private var region = Region.text
  private var buffer = ""
  private var hasSeenListStart = false

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> [EdgeToolsGenerationPart] {
    self.buffer.append(token.stringValue)
    var parts = [EdgeToolsGenerationPart]()
    while let part = self.nextPart() { parts.append(part) }
    return parts
  }

  public mutating func finish() -> [EdgeToolsGenerationPart] {
    defer {
      self.region = .text
      self.buffer = ""
      self.hasSeenListStart = false
    }
    guard !self.buffer.isEmpty else { return [] }
    return switch self.region {
    case .text: [.text(self.buffer)]
    case .tool: [.text("<tool_call>" + self.buffer)]
    }
  }

  private mutating func nextPart() -> EdgeToolsGenerationPart? {
    let opener = "<tool_call>"
    switch self.region {
    case .text:
      return self.nextTextPart(opener: opener)
    case .tool:
      return self.nextToolPart()
    }
  }

  private mutating func nextTextPart(opener: String) -> EdgeToolsGenerationPart? {
    guard let range = opener.firstRange(in: self.buffer[...]) else {
      let retained = self.buffer.suffixPrefixLength(of: opener)
      guard self.buffer.count > retained else { return nil }
      let end = self.buffer.index(self.buffer.endIndex, offsetBy: -retained)
      defer { self.buffer.removeSubrange(..<end) }
      return .text(String(self.buffer[..<end]))
    }
    if range.lowerBound > self.buffer.startIndex {
      let text = String(self.buffer[..<range.lowerBound])
      self.buffer.removeSubrange(..<range.lowerBound)
      return .text(text)
    }
    self.buffer.removeSubrange(..<range.upperBound)
    self.region = .tool
    return self.nextPart()
  }

  private mutating func nextToolPart() -> EdgeToolsGenerationPart? {
    if !self.hasSeenListStart {
      guard let listStart = self.buffer.firstIndex(of: "[") else { return nil }
      self.buffer.removeSubrange(...listStart)
      self.hasSeenListStart = true
    }
    self.buffer.removeLeadingWhitespaceAndCommas()
    if self.buffer.first == "]" {
      self.buffer.removeFirst()
      self.region = .text
      self.hasSeenListStart = false
      return self.nextPart()
    }

    let bytes = Array(self.buffer.utf8)
    guard let objectRange = bytes.firstCompleteJSONObjectRange() else { return nil }
    let object = String(decoding: bytes[objectRange], as: UTF8.self)
    self.buffer.removeSubrange(..<self.buffer.index(self.buffer.startIndex, offsetBy: object.count))
    guard let value = try? EdgeToolsValue(json: object),
      let call = EdgeRawToolCall(jsonValue: value)
    else {
      return .text(object)
    }
    return .toolCall(call)
  }
}

extension String {
  fileprivate mutating func removeLeadingWhitespaceAndCommas() {
    while let first, first.isWhitespace || first == "," {
      self.removeFirst()
    }
  }
}

// MARK: - NeedleGenerateParameters

public protocol NeedleGenerateParameters: EdgeToolsEngineGenerateParameters {
  var toolCallRange: GrammarToolCallRange { get }
}

#if XGrammar
  // MARK: - XGRTokenizerInfo

  extension XGRTokenizerInfo {
    public static func needle(
      tokenizer: some XGRTokenizer,
      vocabularySize: Int = .needleVocabularySize,
      stopTokenIds: Set<EdgeToolsToken.ID> = []
    ) throws -> XGRTokenizerInfo {
      try Self.needle(
        vocabulary: tokenizer.convertIdsToTokens(Array(0..<vocabularySize)),
        eosTokenID: tokenizer.eosTokenId,
        vocabularySize: vocabularySize,
        stopTokenIds: stopTokenIds
      )
    }

    private static func needle(
      vocabulary: [String?],
      eosTokenID: EdgeToolsToken.ID?,
      vocabularySize: Int,
      stopTokenIds: Set<EdgeToolsToken.ID>
    ) throws -> XGRTokenizerInfo {
      guard let eosTokenID, vocabulary.allSatisfy({ $0 != nil }) else {
        throw XGRError(
          code: XGRError.Code.invalidNeedleTokenizer,
          message: "Needle requires a tokenizer with an EOS token and full vocabulary."
        )
      }
      var stopTokenIds = stopTokenIds
      stopTokenIds.insert(eosTokenID)
      return try XGRTokenizerInfo(
        encodedVocabulary: vocabulary.compactMap { $0 },
        vocabularyType: .byteFallback,
        vocabularySize: vocabularySize,
        stopTokenIDs: stopTokenIds.sorted(),
        addPrefixSpace: true
      )
    }
  }

  // MARK: - Grammar

  extension XGRGrammar {
    public static func needle(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange = .unbounded(minimum: 0)
    ) throws -> XGRGrammar {
      let calls = try XGRGrammar.needleCalls(
        tools: tools.map { $0.needleNormalized() },
        range: range
      )
      let prefix = try XGRGrammar.literal("<tool_call> [")
      let prefixedCalls = try prefix.concatenate(calls)
      return try prefixedCalls.concatenate(XGRGrammar.literal("]"))
    }

    private static func needleCalls(
      tools: [EdgeToolDefinition],
      range: GrammarToolCallRange
    ) throws -> XGRGrammar {
      guard range.lowerBound >= 0 else { throw XGRError.invalidToolInvocationRange }
      guard let firstTool = tools.first else { return try XGRGrammar.literal("") }

      var call = try XGRGrammar.needleCall(firstTool)
      for tool in tools.dropFirst() {
        call = try call.union(XGRGrammar.needleCall(tool))
      }
      return try Self.repeatingToolCall(call, separator: ",", range: range)
    }

    private static func needleCall(_ tool: EdgeToolDefinition) throws -> XGRGrammar {
      let schema = tool.arguments.orderedJSONString()
      let arguments = try XGRGrammar.jsonSchema(
        schema,
        configuration: JSONSchemaConfiguration(
          anyWhitespace: false,
          separators: .init(comma: ",", colon: ":"),
          isStrict: true
        )
      )
      let namePrefix = try XGRGrammar.literal("{\"name\":\"")
      let named = try namePrefix.concatenate(XGRGrammar.literal(tool.name))
      let argumentsPrefix = try named.concatenate(XGRGrammar.literal("\",\"arguments\":"))
      let withArguments = try argumentsPrefix.concatenate(arguments)
      return try withArguments.concatenate(XGRGrammar.literal("}"))
    }
  }
#endif
