enum NeedleExportTensorName {
  static let inputIDs = "input_ids"
  static let cachePosition = "cache_position"
  static let selfAttentionMask = "self_attention_mask"
  static let crossAttentionMask = "cross_attention_mask"
  static let encoderProjectedK = "encoder_projected_k"
  static let encoderProjectedV = "encoder_projected_v"
  static let keyCache = "key_cache"
  static let valueCache = "value_cache"
  static let keyCacheDelta = "key_cache_delta"
  static let valueCacheDelta = "value_cache_delta"
  static let logits = "logits"

  static func selfAttentionKeyCache(layer: Int) -> String {
    "self_attention_key_cache_\(layer)"
  }

  static func selfAttentionValueCache(layer: Int) -> String {
    "self_attention_value_cache_\(layer)"
  }

  static func crossAttentionKeyCache(layer: Int) -> String {
    "cross_attention_key_cache_\(layer)"
  }

  static func crossAttentionValueCache(layer: Int) -> String {
    "cross_attention_value_cache_\(layer)"
  }
}
