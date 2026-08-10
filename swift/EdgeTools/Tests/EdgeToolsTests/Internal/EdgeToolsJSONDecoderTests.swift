import CustomDump
import Testing

@testable import EdgeTools

@Suite
struct `EdgeToolsJSONDecoder tests` {
  private struct Configuration: Decodable, Equatable {
    struct Runtime: Decodable, Equatable {
      let threads: Int
      let enabled: Bool
    }

    let name: String
    let dimensions: [Int]
    let runtime: Runtime
    let labels: [String: String]
    let maximumIdentifier: UInt64
    let note: String?
  }

  @Test
  func `Decodes A Model Configuration`() throws {
    let json = #"{"name":"needle","dimensions":[512,1024],"runtime":{"threads":4,"enabled":true},"labels":{"family":"encoder-decoder"},"maximumIdentifier":18446744073709551615,"note":null}"#

    let configuration = try EdgeToolsJSONDecoder().decode(
      Configuration.self,
      from: Array(json.utf8)
    )

    expectNoDifference(
      configuration,
      Configuration(
        name: "needle",
        dimensions: [512, 1024],
        runtime: Configuration.Runtime(threads: 4, enabled: true),
        labels: ["family": "encoder-decoder"],
        maximumIdentifier: UInt64.max,
        note: nil
      )
    )
  }

  @Test
  func `Reports The Coding Path For A Type Mismatch`() {
    let json = #"{"name":"needle","dimensions":[512,"wide"]}"#

    let error = #expect(throws: DecodingError.self) {
      try EdgeToolsJSONDecoder().decode(Configuration.self, from: Array(json.utf8))
    }
    guard case .typeMismatch(_, let context) = error else { return }
    expectNoDifference(context.codingPath.map(\.stringValue), ["dimensions", "Index 1"])
  }
}
