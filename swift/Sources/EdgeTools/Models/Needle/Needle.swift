#if XGrammar
  import EdgeToolsXGrammar
#endif

#if Foundation
  import _EdgeToolsFoundation
#endif

// MARK: - NeedleModelConfiguration

public struct NeedleModelConfiguration: Hashable, Sendable, Codable {
  public var vocabularySize: Int = 8192
  public var dimensions: Int = 512
  public var hiddenDimensions: Int = 512
  public var attentionHeads: Int = 8
  public var kvHeads: Int = 4
  public var encoderLayers: Int = 12
  public var decoderLayers: Int = 8
  public var hiddenLayers: Int = 8
  public var ropeTheta: Float = 10000.0
  public var rmsNormEps: Float = 1e-6
  public var padTokenId: EdgeToolsToken.ID = 0
  public var decoderStartTokenId: EdgeToolsToken.ID = 1
  public var tieWordEmbeddings: Bool = true
  public var maxSeqLen: Int?
  public var maxPositionEmbeddings: Int?
  private var decoderMaxLengthValue: Int?
  private var dtypeValue: String?
  private var torchDTypeValue: String?

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
    vocabularySize: Int = 8192,
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

  private enum CodingKeys: String, CodingKey {
    case vocabularySize = "vocab_size"
    case dimensions = "d_model"
    case hiddenDimensions = "hidden_size"
    case attentionHeads = "num_attention_heads"
    case encoderLayers = "num_encoder_layers"
    case decoderLayers = "num_decoder_layers"
    case hiddenLayers = "num_hidden_layers"
    case kvHeads = "num_kv_heads"
    case ropeTheta = "rope_theta"
    case rmsNormEps = "rms_norm_eps"
    case padTokenId = "pad_token_id"
    case decoderStartTokenId = "decoder_start_token_id"
    case tieWordEmbeddings = "tie_word_embeddings"
    case maxSeqLen = "max_seq_len"
    case maxPositionEmbeddings = "max_position_embeddings"
    case decoderMaxLengthValue = "decoder_max_length"
    case dtypeValue = "dtype"
    case torchDTypeValue = "torch_dtype"
  }
}

// MARK: - NeedleNumerics

enum NeedleNumerics {
  static let float16ClippingMagnitude: Float = 65_500
  static let maskedAttentionScore = -Self.float16ClippingMagnitude
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

// MARK: - NeedleModelInput

public struct NeedleModelInput: Hashable, Sendable {
  public var prompt: NeedlePrompt
  public var tools: [EdgeToolDefinition]
  public var tokenIds: [EdgeToolsToken.ID]

  public init(
    prompt: NeedlePrompt,
    tools: [EdgeToolDefinition],
    tokenIds: [EdgeToolsToken.ID]
  ) {
    self.prompt = prompt
    self.tools = tools
    self.tokenIds = tokenIds
  }
}

// MARK: - Formatting

extension NeedlePrompt {
  public func formatted(tools: [EdgeToolDefinition]) throws -> String {
    let separator = self.system.isEmpty || self.user.isEmpty ? "" : "\n\n"
    let toolsSchema =
      try tools
      .filter(\.includesSchemaInInstructions)
      .map { $0.needleNormalized() }
      .needlePromptEncoded()
    return "\(self.system)\(separator)\(self.user)<tools>\(toolsSchema)"
  }

  #if XGrammar
    public func tokenized(
      tools: [EdgeToolDefinition],
      using tokenizer: some EdgeToolsXGRTokenizer
    ) throws -> [EdgeToolsToken] {
      let tokenIds = tokenizer.encode(text: try self.formatted(tools: tools))
      let tokens = tokenizer.convertIdsToTokens(tokenIds)
      return zip(tokenIds, tokens)
        .compactMap { (tokenId, token) in
          token.map { EdgeToolsToken(id: tokenId, stringValue: $0) }
        }
    }
  #endif
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
    let arguments = self.arguments.orderedKeyEncoded()
    return #"{"name":\#(name),"description":\#(description),"arguments":\#(arguments)}"#
  }
}

extension Sequence where Element == EdgeToolDefinition {
  fileprivate func needlePromptEncoded() throws -> String {
    let definitions = try self.map { try $0.needlePromptEncoded() }
    return "[\(definitions.joined(separator: ","))]"
  }
}

// MARK: - NeedleToolCallParser

public struct NeedleToolCallParser: EdgeToolCallParser, Sendable {
  private var list = IncrementalToolCallList(opener: "<tool_call>")

  public init() {}

  public mutating func accept(token: EdgeToolsToken) -> EdgeRawToolCall? {
    self.list.append(token)
    while let objectData = self.list.nextItem(findRange: { $0.firstCompleteJSONObjectRange() }) {
      if let value = try? EdgeToolsJSONDecoder().decode(EdgeToolsValue.self, from: objectData),
        let call = EdgeRawToolCall(jsonValue: value)
      {
        return call
      }
    }
    return nil
  }
}

// MARK: - Needle Grammar

#if XGrammar
  // MARK: - XGR Tokenizer Info

  extension XGRTokenizerInfo {
    public static func needle(
      tokenizer: some EdgeToolsXGRTokenizer,
      vocabularySize: Int = 8192
    ) throws -> XGRTokenizerInfo {
      try Self.needle(
        vocabulary: tokenizer.convertIdsToTokens(Array(0..<vocabularySize)),
        eosTokenID: tokenizer.eosTokenId,
        vocabularySize: vocabularySize
      )
    }

    private static func needle(
      vocabulary: [String?],
      eosTokenID: EdgeToolsToken.ID?,
      vocabularySize: Int
    ) throws -> XGRTokenizerInfo {
      guard let eosTokenID, vocabulary.allSatisfy({ $0 != nil }) else {
        throw XGRError(
          code: EdgeToolsXGRError.invalidNeedleTokenizer,
          message: "Needle requires a tokenizer with an EOS token and full vocabulary."
        )
      }
      return try XGRTokenizerInfo(
        encodedVocabulary: vocabulary.compactMap { $0 },
        vocabularyType: .byteFallback,
        vocabularySize: vocabularySize,
        stopTokenIDs: [eosTokenID],
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
      let schema = tool.arguments.orderedKeyEncoded()
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
