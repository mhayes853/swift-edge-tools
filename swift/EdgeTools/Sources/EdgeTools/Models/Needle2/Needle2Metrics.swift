#if Needle2
  // MARK: - Metric Keys

  extension EdgeToolsMetricKey {
    public static let needle2ResponseType = Self(rawValue: "Needle2ResponseType")
    public static let needle2PeakRAMMegabytes = Self(rawValue: "Needle2PeakRAMMegabytes")
  }

  // MARK: - Metric Values

  extension EdgeToolsMetrics {
    public var needle2ResponseType: String? {
      get { self[.needle2ResponseType] as? String }
      set { self[.needle2ResponseType] = newValue }
    }

    public var needle2PeakRAMMegabytes: Double? {
      get { self[.needle2PeakRAMMegabytes] as? Double }
      set { self[.needle2PeakRAMMegabytes] = newValue }
    }
  }
#endif
