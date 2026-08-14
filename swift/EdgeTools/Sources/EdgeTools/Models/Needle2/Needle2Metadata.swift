#if Needle2
  // MARK: - Metadata Keys

  extension EdgeToolsMetadataKey {
    public static let needle2ResponseType = Self(rawValue: "Needle2ResponseType")
    public static let needle2PrefillTokensPerSecond =
      Self(rawValue: "Needle2PrefillTokensPerSecond")
    public static let needle2DecodeTokensPerSecond =
      Self(rawValue: "Needle2DecodeTokensPerSecond")
    public static let needle2PeakRAMMegabytes = Self(rawValue: "Needle2PeakRAMMegabytes")
  }

  // MARK: - Metadata Values

  extension EdgeToolsMetadata {
    public var needle2ResponseType: String? {
      get { self[.needle2ResponseType] as? String }
      set { self[.needle2ResponseType] = newValue }
    }

    public var needle2PrefillTokensPerSecond: Double? {
      get { self[.needle2PrefillTokensPerSecond] as? Double }
      set { self[.needle2PrefillTokensPerSecond] = newValue }
    }

    public var needle2DecodeTokensPerSecond: Double? {
      get { self[.needle2DecodeTokensPerSecond] as? Double }
      set { self[.needle2DecodeTokensPerSecond] = newValue }
    }

    public var needle2PeakRAMMegabytes: Double? {
      get { self[.needle2PeakRAMMegabytes] as? Double }
      set { self[.needle2PeakRAMMegabytes] = newValue }
    }
  }
#endif
