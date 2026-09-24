import CustomDump
import EdgeTools
import Testing

@Suite
struct `EdgeToolsStreamGenerable tests` {
  @Test
  func `Private Nested Type Supports Stream Parsing`() throws {
    let partial = try PrivateStreamContainer.Input.Partial(edgeToolsValue: ["value": "hello"])
    expectNoDifference(partial.value.map(String.init), "hello")
  }

  @Test
  func `Public Partial Uses An Optional Field Schema`() throws {
    let partial = try PublicStreamPayload.Partial(edgeToolsValue: [:])
    expectNoDifference(partial.text == nil, true)
    expectNoDifference(
      PublicStreamPayload.Partial.edgeToolsGenerationSchema.objectValue?[.required],
      nil
    )
  }

  @Test
  func `Partial JSON Round Trips While Generation Is Incomplete`() throws {
    let schema = StreamSearchRequest.Partial.edgeToolsGenerationSchema
    let properties = try #require(schema.objectValue?[.properties])
    let fields = try _edgeToolsRequireObjectValue(properties)
    expectNoDifference(fields["q"] != nil, true)
    expectNoDifference(schema.objectValue?[.required], nil)

    var parser = PartialsStream(initialValue: StreamSearchRequest.Partial(), from: .json())
    try parser.next("{\"q\":\"hel".utf8)

    let snapshot = parser.current.edgeToolsValue
    let restored = try StreamSearchRequest.Partial(edgeToolsValue: snapshot)
    expectNoDifference(restored.edgeToolsValue, snapshot)
    expectNoDifference(restored.query.map { String($0) }, "hel")
    expectNoDifference(StreamSearchRequest(streamPartial: restored) == nil, true)

    try parser.next("lo\",\"scores\":[1,2]}".utf8)
    let complete = try parser.finish()
    let response = try #require(StreamSearchRequest(streamPartial: complete))
    expectNoDifference(response.query, "hello")
    expectNoDifference(response.scores, [1, 2])
    expectNoDifference(response.cacheHits, 0)
    expectNoDifference(response.requestID, nil)
  }

  @Test
  func `Nested Partial Round Trips`() throws {
    var parser = PartialsStream(initialValue: StreamPerson.Partial(), from: .json())
    try parser.next("{\"address\":{\"city\":\"Pa".utf8)

    let snapshot = parser.current.edgeToolsValue
    let restored = try StreamPerson.Partial(edgeToolsValue: snapshot)
    expectNoDifference(restored.edgeToolsValue, snapshot)
    expectNoDifference(restored.address?.city.map { String($0) }, "Pa")
  }

  @Test
  func `Open Array Elements And Dictionary Values Round Trip`() throws {
    var parser = PartialsStream(initialValue: StreamCollections.Partial(), from: .json())
    try parser.next("{\"tags\":[\"al".utf8)
    expectNoDifference(parser.current.edgeToolsValue, ["tags": ["al"]])

    try parser.next("pha\"],\"values\":{\"count\":\"tw".utf8)
    let snapshot = parser.current.edgeToolsValue
    expectNoDifference(snapshot, ["tags": ["alpha"], "values": ["count": "tw"]])
    let restored = try StreamCollections.Partial(edgeToolsValue: snapshot)
    expectNoDifference(restored.edgeToolsValue, snapshot)
  }

  @Test
  func `Enum Partial Round Trips`() throws {
    var parser = PartialsStream(initialValue: StreamAction.Partial(), from: .json())
    try parser.next("{\"search\":{\"query\":\"abc".utf8)

    let snapshot = parser.current.edgeToolsValue
    let restored = try StreamAction.Partial(edgeToolsValue: snapshot)
    expectNoDifference(restored.edgeToolsValue, snapshot)

    try parser.next("\"}}".utf8)
    let complete = try parser.finish()
    let action = try #require(StreamAction(streamPartial: complete))
    expectNoDifference(action, .search(query: "abc", limit: nil))
  }

}

private struct PrivateStreamContainer {
  @EdgeToolsGenerable
  struct Input {
    var value: String
  }
}

@EdgeToolsGenerable
private struct StreamSearchRequest {
  @EdgeToolsGuide(key: "q")
  var query: String

  var scores: [Int]

  @EdgeToolsIgnored
  var cacheHits: Int = 0

  @EdgeToolsIgnored
  var requestID: String?
}

@EdgeToolsGenerable
private struct StreamAddress {
  var city: String
}

@EdgeToolsGenerable
private struct StreamPerson {
  var address: StreamAddress
}

@EdgeToolsGenerable
private struct StreamCollections {
  var tags: [String]
  var values: [String: String]
}

@EdgeToolsGenerable
private enum StreamAction: Equatable {
  @StreamParseableDefault
  case idle(reason: String?)

  case search(query: String, limit: Int?)
}

@EdgeToolsGenerable
public struct PublicStreamPayload {
  public var text: String
}
