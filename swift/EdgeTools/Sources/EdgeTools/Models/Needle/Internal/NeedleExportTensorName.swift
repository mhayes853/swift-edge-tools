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
}
