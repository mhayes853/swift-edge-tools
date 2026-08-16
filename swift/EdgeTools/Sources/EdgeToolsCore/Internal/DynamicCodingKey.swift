#if !$Embedded
  package struct DynamicCodingKey: CodingKey, Hashable, Sendable {
    package let stringValue: String
    package let intValue: Int?

    package init(_ stringValue: String) {
      self.stringValue = stringValue
      self.intValue = nil
    }

    package init?(stringValue: String) {
      self.init(stringValue)
    }

    package init?(intValue: Int) {
      self.stringValue = String(intValue)
      self.intValue = intValue
    }
  }
#endif
