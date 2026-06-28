// MARK: - NeedleModelConfiguration

public struct NeedleModelConfiguration: Hashable, Sendable {
  public let vocabularySize: Int
  public let dimensions: Int
  public let hiddenDimensions: Int
  public let attentionHeads: Int
  public let kvHeads: Int
  public let encoderLayers: Int
  public let decoderLayers: Int
  public let hiddenLayers: Int
  public let ropeTheta: Float
  public let rmsNormEps: Float
  public let padTokenId: NeedleToken.ID
  public let decoderStartTokenId: NeedleToken.ID
  public var tieWordEmbeddings: Bool
  private let maxSeqLen: Int?
  private let maxPositionEmbeddings: Int?
  private let _dtype: String?

  public var dtype: String {
    self._dtype ?? "bfloat16"
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
}

// MARK: - Codable

extension NeedleModelConfiguration: Codable {
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
    case _dtype = "dtype"
  }
}
