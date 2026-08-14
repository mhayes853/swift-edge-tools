#if Needle2
  import OrderedCollections

  #if Foundation
    import _EdgeToolsFoundation
  #endif

  #if JS && canImport(JavaScriptKit)
    import JavaScriptKit
    import _EdgeToolsJavaScript
  #endif

  #if canImport(Network)
    import Network
  #endif

  #if os(macOS) && canImport(IOKit.ps)
    import IOKit.ps
  #endif

  #if os(Windows) && canImport(WinSDK)
    import WinSDK
  #endif

  #if os(watchOS) && canImport(WatchKit)
    import WatchKit
  #elseif canImport(UIKit)
    import UIKit
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
      let recognizedKeys: [Key] = [
        .date, .locale, .device, .battery,
        .network, .location, .user, .assistant
      ]
      let recognized = recognizedKeys.compactMap { key in
        self.values[key].map { (key, $0) }
      }
      let custom = self.values.filter { !recognizedKeys.contains($0.key) }
      return (recognized + custom)
        .map { key, value in
          "\(key.rawValue.split(separator: ";", omittingEmptySubsequences: false).joined(separator: " ")): \(value)"
        }
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
      public static let tablet: Self = "tablet"
      public static let desktop: Self = "desktop"
      public static let tv: Self = "tv"
      public static let watch: Self = "watch"
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

  // MARK: - Foundation

  #if Foundation
    extension Needle2System {
      public static func date(_ date: Date, timeZone: TimeZone = .current) -> Self {
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

  extension Needle2System {
    public static func platformDefaults(
      device: (@Sendable () async -> Device?)? = nil,
      battery: (@Sendable () async -> Float?)? = nil,
      network: (@Sendable () async -> Network?)? = nil,
      location: (@Sendable () async -> String?)? = nil
    ) async -> Self {
    #if JS && canImport(JavaScriptKit) && !Foundation
      async let deviceValue = device?()
      async let batteryValue = battery?()
      async let networkValue = network?()
      async let locationValue = location?()

      let options = JSObject()
      let overrides = JSObject()
      if device != nil {
        overrides[Key.device.rawValue] = await deviceValue.map { .string($0.rawValue) } ?? .null
      }
      if battery != nil {
        overrides[Key.battery.rawValue] = await batteryValue.map {
          .string("\(formattedBatteryPercentage($0))%")
        } ?? .null
      }
      if network != nil {
        overrides[Key.network.rawValue] = await networkValue.map { .string($0.rawValue) } ?? .null
      }
      if location != nil {
        overrides[Key.location.rawValue] = await locationValue.map { .string($0) } ?? .null
      }
      options["overrides"] = overrides.jsValue

      let defaults: Self?
      if let object = try? await _EdgeToolsJavaScript.needle2DefaultSystemValues(options) {
        defaults = Self.javascriptSystem(from: object)
      } else {
        defaults = nil
      }
      var system = defaults ?? Self(OrderedDictionary<Key, String>())
      guard defaults == nil else {
        return system
      }
      if let device = await deviceValue {
        system.merge(.device(device))
      }
      if let battery = await batteryValue {
        system.merge(.battery(percent: battery))
      }
      if let network = await networkValue {
        system.merge(.network(network))
      }
      if let location = await locationValue {
        system.merge(.location(location))
      }
      return system
    #else
      let deviceProvider = device ?? { await Needle2System.platformDevice() }
      let batteryProvider = battery ?? { await Needle2System.platformBatteryPercentage() }
      let networkProvider = network ?? { await Needle2System.platformNetwork() }
      let locationProvider = location ?? { await Needle2System.platformLocation() }

      async let device = deviceProvider()
      async let battery = batteryProvider()
      async let network = networkProvider()
      async let location = locationProvider()

      #if Foundation
        var system: Self = [.date(Date()), .locale(Locale.current)]
      #else
        var system = Self(OrderedDictionary<Key, String>())
      #endif
      if let device = await device {
        system.merge(.device(device))
      }
      if let battery = await battery {
        system.merge(.battery(percent: battery))
      }
      if let network = await network {
        system.merge(.network(network))
      }
      if let location = await location {
        system.merge(.location(location))
      }
      return system
    #endif
    }

    public static func platformDevice() async -> Device? {
      await Needle2PlatformDefaults.device
    }

    public static func platformBatteryPercentage() async -> Float? {
      await Needle2PlatformDefaults.batteryPercentage()
    }

    public static func platformNetwork() async -> Network? {
    #if canImport(Network) || os(Linux) || os(Android)
      await Needle2PlatformDefaults.network
    #else
      Needle2PlatformDefaults.network
    #endif
    }

    public static func platformLocation() async -> String? {
      Needle2PlatformDefaults.location
    }
  }

  #if JS && canImport(JavaScriptKit) && !Foundation
    extension Needle2System {
      private static func javascriptSystem(from object: JSObject) -> Self {
        let keys: [Key] = [.date, .locale, .device, .battery, .network, .location]
        return Self(OrderedDictionary(uniqueKeysWithValues: keys.compactMap { key in
          object[key.rawValue].stringValue.map { (key, $0) }
        }))
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

  #if Foundation
    private let needle2DateFormatter = Lock(makeNeedle2DateFormatter())

    private func makeNeedle2DateFormatter() -> DateFormatter {
      let formatter = DateFormatter()
      formatter.calendar = Calendar(identifier: .gregorian)
      formatter.locale = Locale(identifier: "en_US_POSIX")
      formatter.dateFormat = "yyyy-MM-dd EEE HH:mm"
      return formatter
    }
  #endif

  // MARK: - Platform Defaults

  enum Needle2PlatformDefaults {
    static var device: Needle2System.Device? {
      get async {
      #if os(watchOS)
        await MainActor.run { .watch }
      #elseif os(macOS) || os(Linux) || os(Windows)
        .desktop
      #elseif canImport(UIKit)
        await MainActor.run {
          switch UIDevice.current.userInterfaceIdiom {
          case .phone: .phone
          case .pad: .tablet
          case .tv: .tv
          default: nil
          }
        }
      #else
        nil
      #endif
      }
    }

    static func batteryPercentage() async -> Float? {
      #if os(watchOS) && canImport(WatchKit)
        await MainActor.run {
          let device = WKInterfaceDevice.current()
          let wasMonitoringEnabled = device.isBatteryMonitoringEnabled
          device.isBatteryMonitoringEnabled = true
          defer { device.isBatteryMonitoringEnabled = wasMonitoringEnabled }
          let level = device.batteryLevel
          guard (0...1).contains(level) else { return nil }
          return level * 100
        }
      #elseif canImport(UIKit) && !os(tvOS)
        await MainActor.run {
          let device = UIDevice.current
          let wasMonitoringEnabled = device.isBatteryMonitoringEnabled
          device.isBatteryMonitoringEnabled = true
          defer { device.isBatteryMonitoringEnabled = wasMonitoringEnabled }
          let level = device.batteryLevel
          guard (0...1).contains(level) else { return nil }
          return level * 100
        }
      #elseif os(macOS) && canImport(IOKit.ps)
        Self.macOSBatteryPercentage()
      #elseif os(Linux) || os(Android)
        Self.unixBatteryPercentage()
      #elseif os(Windows)
        Self.windowsBatteryPercentage()
      #else
        nil
      #endif
    }

    #if os(macOS) && canImport(IOKit.ps)
      private static func macOSBatteryPercentage() -> Float? {
        guard let snapshot = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
          let sources = IOPSCopyPowerSourcesList(snapshot)?.takeRetainedValue() as? [CFTypeRef]
        else {
          return nil
        }
        for source in sources {
          guard
            let description = IOPSGetPowerSourceDescription(snapshot, source)?.takeUnretainedValue() as? [String: Any],
            description[kIOPSTypeKey as String] as? String == kIOPSInternalBatteryType as String,
            let current = description[kIOPSCurrentCapacityKey as String] as? Int,
            let maximum = description[kIOPSMaxCapacityKey as String] as? Int,
            maximum > 0
          else {
            continue
          }
          return Float(current) / Float(maximum) * 100
        }
        return nil
      }
    #endif

    #if canImport(Network)
      static var network: Needle2System.Network? {
        get async {
          let monitor = NWPathMonitor()
          return await withUnsafeContinuation { continuation in
            monitor.pathUpdateHandler = { path in
              let network: Needle2System.Network?
              if path.status != .satisfied {
                network = .offline
              } else if path.usesInterfaceType(.wifi) {
                network = .wifi
              } else if path.usesInterfaceType(.cellular) {
                network = .cellular
              } else if path.usesInterfaceType(.wiredEthernet) {
                network = .ethernet
              } else {
                network = nil
              }
              monitor.cancel()
              continuation.resume(returning: network)
            }
            monitor.start(queue: .global(qos: .utility))
          }
        }
      }
    #elseif os(Linux) || os(Android)
      static var network: Needle2System.Network? {
        get async { Self.unixNetwork() }
      }
    #else
      static var network: Needle2System.Network? { nil }
    #endif

    static var location: String? { nil }

    #if os(Linux) || os(Android)
      static func unixBatteryPercentage() -> Float? {
        let root = "/sys/class/power_supply"
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: root) else {
          return nil
        }

        let percentages = names
          .filter { name in
            let type = powerSupplyText(name, key: "type", root: root)
            return type == "Battery" || name.hasPrefix("BAT") || name.lowercased() == "battery"
          }
          .compactMap { batteryPercentage(name: $0, root: root) }

        guard !percentages.isEmpty else { return nil }
        return percentages.reduce(0, +) / Float(percentages.count)
      }

      private static func batteryPercentage(name: String, root: String) -> Float? {
        if let percentage = powerSupplyValue(name, key: "capacity", root: root),
          (0...100).contains(percentage) {
          return percentage
        }

        return [("energy_now", "energy_full"), ("charge_now", "charge_full")]
          .compactMap { currentKey, maximumKey in
            guard let current = powerSupplyValue(name, key: currentKey, root: root),
              let maximum = powerSupplyValue(name, key: maximumKey, root: root),
              maximum > 0
            else {
              return nil
            }
            let percentage = current / maximum * 100
            return (0...100).contains(percentage) ? percentage : nil
          }
          .first
      }

      private static func powerSupplyText(_ name: String, key: String, root: String) -> String? {
        let path = "\(root)/\(name)/\(key)"
        guard let contents = try? String(contentsOfFile: path, encoding: .utf8) else { return nil }
        return contents.trimmingCharacters(in: .whitespacesAndNewlines)
      }

      private static func powerSupplyValue(_ name: String, key: String, root: String) -> Float? {
        powerSupplyText(name, key: key, root: root).flatMap(Float.init)
      }

      static func unixNetwork() -> Needle2System.Network? {
        let root = "/sys/class/net"
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: root) else {
          return .offline
        }

        let interface = defaultRouteInterface()
          ?? names.first(where: { isActiveInterface($0, root: root) })
        guard let interface else { return .offline }
        return network(for: interface)
      }

      private static func defaultRouteInterface() -> String? {
        guard let contents = try? String(contentsOfFile: "/proc/net/route", encoding: .utf8) else {
          return nil
        }

        return contents
          .split(whereSeparator: \.isNewline)
          .dropFirst()
          .compactMap { line in
            let fields = line.split(whereSeparator: { $0 == " " || $0 == "\t" })
            guard fields.count > 1, fields[1] == "00000000" else { return nil }
            return String(fields[0])
          }
          .first
      }

      private static func isActiveInterface(_ name: String, root: String) -> Bool {
        guard name != "lo" else { return false }
        return (try? String(
          contentsOfFile: "\(root)/\(name)/operstate",
          encoding: .utf8
        ))?.trimmingCharacters(in: .whitespacesAndNewlines) == "up"
      }

      private static func network(for interface: String) -> Needle2System.Network? {
        let name = interface.lowercased()
        switch name {
        case _ where name.contains("wlan") || name.contains("wifi"):
          .wifi
        case _ where name.contains("rmnet") || name.contains("ccmni") || name.contains("wwan"):
          .cellular
        case _ where name.hasPrefix("eth") || name.hasPrefix("en"):
          .ethernet
        default:
          nil
        }
      }
    #endif

    #if os(Windows) && canImport(WinSDK)
      static func windowsBatteryPercentage() -> Float? {
        var status = SYSTEM_POWER_STATUS()
        guard GetSystemPowerStatus(&status) != FALSE else { return nil }

        let percentage = Int(status.BatteryLifePercent)
        guard (0...100).contains(percentage) else { return nil }
        return Float(percentage)
      }
    #elseif os(Windows)
      static func windowsBatteryPercentage() -> Float? {
        nil
      }
    #endif
  }
#endif
