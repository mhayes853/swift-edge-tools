#if Needle2
  import CustomDump
  import EdgeTools
  import Testing

  #if FullFoundation
    import Foundation
  #endif

  @Suite
  struct `Needle2System tests` {
    @Test
    func `Formats Recognized And Custom Facts`() {
      let system: Needle2System = [
        .date("2026-07-21 Tue 14:30"),
        .locale("en-US"),
        .device(.phone),
        .battery(percent: 62),
        .network(.wifi),
        .location("San Francisco, CA"),
        .user("Matthew"),
        .assistant("Edge Assistant"),
        .raw("temperature-unit", "celsius")
      ]

      expectNoDifference(
        system.formatted(),
        "date: 2026-07-21 Tue 14:30; locale: en-US; device: phone; battery: 62%; network: wifi; location: San Francisco, CA; user: Matthew; assistant: Edge Assistant; temperature-unit: celsius"
      )
    }

    @Test
    func `Merges Facts In Place`() {
      let system = Needle2System(
        [.locale: "en-US", .device: "phone"],
        [.locale: "en-GB", .battery: "80%"]
      )

      expectNoDifference(
        system.formatted(),
        "locale: en-GB; device: phone; battery: 80%"
      )
    }

    @Test
    func `Formats Battery Fractions And Percentages`() {
      expectNoDifference(Needle2System.battery(0.62).formatted(), "battery: 62%")
      expectNoDifference(
        Needle2System.battery(percent: 62.5).formatted(),
        "battery: 62.5%"
      )
    }

    #if FullFoundation
      @Test
      func `Formats Platform Defaults`() async {
        let system = await Needle2System.platformDefaults()

        expectNoDifference(system[.date] != nil, true)
        expectNoDifference(system[.locale], Locale.current.identifier)
        expectNoDifference(system.formatted().hasPrefix("date: "), true)
      }

      @Test
      func `Overrides Platform Default Providers`() async {
        let system = await Needle2System.platformDefaults(
          device: { nil },
          battery: { 62.5 },
          network: { nil }
        )

        expectNoDifference(system[.device], nil)
        expectNoDifference(system[.battery], "62.5%")
        expectNoDifference(system[.network], nil)
      }
    #endif

    @Test
    func `Formats Facts In TypeScript Order And Sanitizes Keys`() {
      let system: Needle2System = [
        .assistant("Needle"),
        .raw(.init(rawValue: "custom;key"), "value"),
        .date("today")
      ]

      expectNoDifference(
        system.formatted(),
        "date: today; assistant: Needle; custom key: value"
      )
    }

    #if FullFoundation
      @Test
      func `Formats Foundation Date And Locale`() {
        let date = Date(timeIntervalSince1970: 1_774_184_400)
        let system: Needle2System = [
          .date(date, timeZone: TimeZone(secondsFromGMT: 0)!),
          .locale(Locale(identifier: "en-US"))
        ]

        expectNoDifference(
          system.formatted(),
          "date: 2026-03-22 Sun 13:00; locale: en-US"
        )
      }
    #endif
  }
#endif
