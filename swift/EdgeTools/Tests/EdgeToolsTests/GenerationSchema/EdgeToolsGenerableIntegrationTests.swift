import CustomDump
import EdgeTools
import Testing

@Suite
struct `EdgeToolsGenerableIntegration tests` {
  @Test
  func `Qualified Property Macros Preserve Package Access`() throws {
    let payload = try PackageQualifiedPayload(edgeToolsValue: ["renamed": "blob"])

    expectNoDifference(payload.value, "blob")
    expectNoDifference(payload.cache, 0)
  }

  @Test
  func `Optional Enum Associated Values May Be Omitted`() throws {
    let action = try OptionalAssociatedValue(
      edgeToolsValue: ["search": ["query": "blob"]]
    )

    expectNoDifference(action, .search(query: "blob", limit: nil))
  }

  @Test
  func `Enum Payloads May Omit Every Optional Value`() throws {
    let action = try OptionalAssociatedValue(edgeToolsValue: ["clear": [:]])

    expectNoDifference(action, .clear(reason: nil))
  }
}

@EdgeToolsGenerable
package struct PackageQualifiedPayload: Equatable {
  @EdgeTools.EdgeToolsGuide(key: "renamed")
  package var value: String

  @EdgeTools.EdgeToolsIgnored
  package var cache: Int = 0
}

@EdgeToolsGenerable
private enum OptionalAssociatedValue: Equatable {
  case search(query: String, limit: Int?)
  case clear(reason: String?)
}
