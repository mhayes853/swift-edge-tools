// MARK: - NeedleMetadataKey

extension NeedleMetadataKey {
  public static let generationConfidence =
    NeedleMetadataKey(rawValue: "GenerationConfidence")

  public static let perTokenConfidences =
    NeedleMetadataKey(rawValue: "PerTokenConfidences")
}

// MARK: - NeedleMetadata

extension NeedleMetadata {
  public var generationConfidence: Float? {
    get { self[.generationConfidence] as? Float }
    set { self[.generationConfidence] = newValue }
  }

  public var perTokenConfidences: [Float]? {
    get { self[.perTokenConfidences] as? [Float] }
    set { self[.perTokenConfidences] = newValue }
  }
}