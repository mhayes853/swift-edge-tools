#if Needle2
  import OrderedCollections

  #if FullFoundation
    import _EdgeToolsFoundation
  #endif

  // MARK: - Needle2System

  public struct Needle2System: Hashable, Sendable {
    public struct Key: Hashable, Sendable, RawRepresentable, ExpressibleByStringLiteral {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public init(stringLiteral value: String) {
        self.init(rawValue: value)
      }
    }

    public var values: OrderedDictionary<Key, String>

    public init(_ values: OrderedDictionary<Key, String>) {
      self.values = values
    }

    public init(_ systems: Self...) {
      self.init(systems)
    }

    public init(_ systems: some Sequence<Self>) {
      self.values = systems.reduce(into: [:]) { values, system in
        for (key, value) in system.values {
          values[key] = value
        }
      }
    }

    public subscript(key: Key) -> String? {
      get { self.values[key] }
      set { self.values[key] = newValue }
    }

    public func merging(_ other: Self) -> Self {
      Self(self, other)
    }

    public mutating func merge(_ other: Self) {
      self = self.merging(other)
    }

    public func formatted() -> String {
      self.values
        .map { "\($0.key.rawValue): \($0.value)" }
        .joined(separator: "; ")
    }
  }

  // MARK: - Literals

  extension Needle2System: ExpressibleByArrayLiteral {
    public init(arrayLiteral elements: Self...) {
      self.init(elements)
    }
  }

  extension Needle2System: ExpressibleByDictionaryLiteral {
    public init(dictionaryLiteral elements: (Key, String)...) {
      self.init(OrderedDictionary(uniqueKeysWithValues: elements))
    }
  }

  // MARK: - Recognized Keys

  extension Needle2System.Key {
    public static let date: Self = "date"
    public static let locale: Self = "locale"
    public static let device: Self = "device"
    public static let battery: Self = "battery"
    public static let network: Self = "network"
    public static let location: Self = "location"
    public static let user: Self = "user"
    public static let assistant: Self = "assistant"
  }

  // MARK: - Raw Helpers

  extension Needle2System {
    public static func raw(_ key: Key, _ value: String) -> Self {
      Self([key: value])
    }

    public static func date(_ value: String) -> Self {
      Self([.date: value])
    }

    public static func locale(_ value: String) -> Self {
      Self([.locale: value])
    }

    public static func location(_ value: String) -> Self {
      Self([.location: value])
    }

    public static func user(_ value: String) -> Self {
      Self([.user: value])
    }

    public static func assistant(_ value: String) -> Self {
      Self([.assistant: value])
    }
  }

  // MARK: - Device

  extension Needle2System {
    public struct Device: Hashable, Sendable, RawRepresentable, ExpressibleByStringLiteral {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public init(stringLiteral value: String) {
        self.init(rawValue: value)
      }

      public static let phone: Self = "phone"
    }

    public static func device(_ value: Device) -> Self {
      Self([.device: value.rawValue])
    }
  }

  // MARK: - Battery

  extension Needle2System {
    public static func battery(_ fraction: Float) -> Self {
      precondition((0...1).contains(fraction), "A battery fraction must be between 0 and 1.")
      return Self.battery(percent: fraction * 100)
    }

    public static func battery(percent: Float) -> Self {
      precondition((0...100).contains(percent), "A battery percentage must be between 0 and 100.")
      return Self([.battery: "\(formattedBatteryPercentage(percent))%"])
    }
  }

  // MARK: - Network

  extension Needle2System {
    public struct Network: Hashable, Sendable, RawRepresentable, ExpressibleByStringLiteral {
      public let rawValue: String

      public init(rawValue: String) {
        self.rawValue = rawValue
      }

      public init(stringLiteral value: String) {
        self.init(rawValue: value)
      }

      public static let wifi: Self = "wifi"
      public static let cellular: Self = "cellular"
      public static let ethernet: Self = "ethernet"
      public static let offline: Self = "offline"
    }

    public static func network(_ value: Network) -> Self {
      Self([.network: value.rawValue])
    }
  }

  // MARK: - Full Foundation

  #if FullFoundation
    extension Needle2System {
      public static func date(
        _ date: Date,
        timeZone: TimeZone = .current
      ) -> Self {
        needle2DateFormatter.withLock { formatter in
          formatter.timeZone = timeZone
          return Self.date(formatter.string(from: date))
        }
      }

      public static func locale(_ locale: Locale) -> Self {
        Self.locale(locale.identifier)
      }
    }
  #endif

  private func formattedBatteryPercentage(_ percentage: Float) -> String {
    let rounded = (percentage * 1_000).rounded() / 1_000
    guard rounded != rounded.rounded() else {
      return String(Int(rounded))
    }
    return String(rounded)
  }

  #if FullFoundation
    private let needle2DateFormatter = Lock(makeNeedle2DateFormatter())

    private func makeNeedle2DateFormatter() -> DateFormatter {
      let formatter = DateFormatter()
      formatter.calendar = Calendar(identifier: .gregorian)
      formatter.locale = Locale(identifier: "en_US_POSIX")
      formatter.dateFormat = "yyyy-MM-dd EEE HH:mm"
      return formatter
    }
  #endif
#endif
