enum NeedleExportTensorName {
  static let inputIDs = "input_ids"
  static let cachePosition = "cache_position"
  static let selfAttentionMask = "self_attention_mask"
  static let crossAttentionMask = "cross_attention_mask"
  static let encoderProjectedK = "encoder_projected_k"
  static let encoderProjectedV = "encoder_projected_v"
  static let keyCache = "key_cache"
  static let valueCache = "value_cache"
  static let updatedKeyCache = "updated_key_cache"
  static let updatedValueCache = "updated_value_cache"
  static let logits = "logits"
}
