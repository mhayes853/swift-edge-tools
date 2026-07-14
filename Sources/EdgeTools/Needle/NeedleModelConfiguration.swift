import Foundation

// MARK: - NeedleModelConfiguration

public struct NeedleModelConfiguration: Hashable, Sendable {
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
  private var _dtype: String?
  private var _torchDType: String?

  public var dtype: String {
    get { self._dtype ?? self._torchDType ?? "bfloat16" }
    set { self._dtype = newValue }
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
    case _torchDType = "torch_dtype"
  }
}

// MARK: - Loading

extension NeedleModelConfiguration {
  static func decode(in directory: URL, decoder: JSONDecoder = JSONDecoder()) throws -> Self? {
    let configurationURLs = [
      directory.appending(path: "configuration.json"),
      directory.appending(path: "config.json")
    ]
    for url in configurationURLs where FileManager.default.fileExists(atPath: url.path()) {
      return try decoder.decode(Self.self, from: Data(contentsOf: url))
    }
    return nil
  }
}
