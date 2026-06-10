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
  public let padTokenId: Int
  public let bosTokenId: Int
  public let eosTokenId: Int
  public let unkTokenId: Int
  public let decoderStartTokenId: Int
  public let dtype: String

  public var attentionHeadDimensions: Int {
    self.dimensions / self.attentionHeads
  }

  public var kvDimensions: Int {
    self.kvHeads * self.attentionHeadDimensions
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
    case bosTokenId = "bos_token_id"
    case eosTokenId = "eos_token_id"
    case unkTokenId = "unk_token_id"
    case decoderStartTokenId = "decoder_start_token_id"
    case dtype = "dtype"
  }
}
