// MARK: - EdgeToolsMetadataKey

extension EdgeToolsMetadataKey {
  public static let generationConfidence =
    EdgeToolsMetadataKey(rawValue: "GenerationConfidence")

  public static let perTokenConfidences =
    EdgeToolsMetadataKey(rawValue: "PerTokenConfidences")
}

// MARK: - EdgeToolsMetadata

extension EdgeToolsMetadata {
  public var generationConfidence: Float? {
    get { self[.generationConfidence] as? Float }
    set { self[.generationConfidence] = newValue }
  }

  public var perTokenConfidences: [Float]? {
    get { self[.perTokenConfidences] as? [Float] }
    set { self[.perTokenConfidences] = newValue }
  }
}
