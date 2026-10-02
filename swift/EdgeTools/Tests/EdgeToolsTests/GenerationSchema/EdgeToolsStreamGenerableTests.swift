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

  @Test
  func `Swift String Partials Preserve Snapshots And Round Trip`() throws {
    var parser = PartialsStream(initialValue: StringStorageRequest.Partial(), from: .json())
    try parser.next(#"{"title":"hé"#.utf8)
    let early = parser.current
    let title: String? = early.title
    expectNoDifference(title, "hé")

    try parser.next(
      #"llo","tags":["a"],"groups":{"g":["b"]},"body":"text","address":{"city":"Paris"}}"#.utf8
    )
    let partial = try parser.finish()
    expectNoDifference(early.title, "hé")
    let tags: StreamArray<String>? = partial.tags
    let groups: StreamDictionary<StreamArray<String>>? = partial.groups
    let body: StreamString? = partial.body
    let city: StreamString? = partial.address?.city
    expectNoDifference(tags.map(Array.init), ["a"])
    expectNoDifference(groups?["g"].map(Array.init), ["b"])
    expectNoDifference(body.map(String.init), "text")
    expectNoDifference(city.map(String.init), "Paris")

    let value = try #require(StringStorageRequest(partial))
    expectNoDifference(value.title, "héllo")
    expectNoDifference(value.streamPartialValue.edgeToolsValue, partial.edgeToolsValue)
    let restored = try StringStorageRequest.Partial(edgeToolsValue: partial.edgeToolsValue)
    expectNoDifference(restored.edgeToolsValue, partial.edgeToolsValue)
  }

  @Test
  func `Member String Storage Overrides The Default`() throws {
    var parser = PartialsStream(initialValue: MemberStringStorage.Partial(), from: .json())
    try parser.next(#"{"text":"hello","other":"world"}"#.utf8)
    let partial = try parser.finish()
    let text: String? = partial.text
    let other: StreamString? = partial.other
    expectNoDifference(text, "hello")
    expectNoDifference(other.map(String.init), "world")
    expectNoDifference(MemberStringStorage(partial)?.text, "hello")
  }

  @Test
  func `Key Strategies Agree Across Schemas Values And Streaming`() throws {
    let expected: EdgeToolsValue = ["user_id": 7, "display_name": "hello", "ExactName": "fixed"]
    let value = try SnakeCaseRequest(edgeToolsValue: expected)
    expectNoDifference(value.edgeToolsValue, expected)
    expectNoDifference(try SnakeCaseRequest(edgeToolsValue: expected).userID, 7)
    expectNoDifference(
      SnakeCaseRequest.edgeToolsGenerationSchema.objectValue?[.required],
      ["user_id", "display_name", "ExactName"]
    )

    var parser = PartialsStream(initialValue: SnakeCaseRequest.Partial(), from: .json())
    try parser.next(#"{"user_id":7,"display_name":"hello","ExactName":"fixed"}"#.utf8)
    let partial = try parser.finish()
    expectNoDifference(partial.edgeToolsValue, expected)
    expectNoDifference(SnakeCaseRequest(partial)?.edgeToolsValue, expected)
    expectNoDifference(
      try SnakeCaseRequest.Partial(edgeToolsValue: expected).edgeToolsValue,
      expected
    )
  }

  @Test
  func `Custom Keys Apply To Enum Cases And Labels But Preserve Positional Keys`() throws {
    var parser = PartialsStream(initialValue: CustomKeyAction.Partial(), from: .json())
    try parser.next(#"{"x_send_message":{"x_body_text":"hi","_1":"there"}}"#.utf8)
    let partial = try parser.finish()
    let expected: EdgeToolsValue = ["x_send_message": ["x_body_text": "hi", "_1": "there"]]
    let action = try #require(CustomKeyAction(partial))
    expectNoDifference(action, .sendMessage(bodyText: "hi", "there"))
    expectNoDifference(action.edgeToolsValue, expected)
    expectNoDifference(partial.edgeToolsValue, expected)
    expectNoDifference(try CustomKeyAction(edgeToolsValue: expected), action)
    expectNoDifference(
      try CustomKeyAction.Partial(edgeToolsValue: expected).edgeToolsValue,
      expected
    )
    expectNoDifference(action.streamPartialValue.edgeToolsValue, expected)
  }

  @Test
  func `Custom Keys Preserve Explicit Guides And Underscored Labels`() throws {
    var parser = PartialsStream(initialValue: CustomKeyRequest.Partial(), from: .json())
    try parser.next(#"{"x_user_id":3,"fixed":"hello"}"#.utf8)
    let partial = try parser.finish()
    let expected: EdgeToolsValue = ["x_user_id": 3, "fixed": "hello"]
    expectNoDifference(partial.edgeToolsValue, expected)
    expectNoDifference(CustomKeyRequest(partial)?.edgeToolsValue, expected)
    expectNoDifference(
      try CustomKeyRequest.Partial(edgeToolsValue: expected).edgeToolsValue,
      expected
    )

    let action = CustomKeyAction.underscored(_bodyText: "hello")
    let enumValue: EdgeToolsValue = ["x_underscored": ["x__body_text": "hello"]]
    expectNoDifference(action.edgeToolsValue, enumValue)
    expectNoDifference(action.streamPartialValue.edgeToolsValue, enumValue)
  }

  @Test
  func `Configured Cache Owns Struct And Enum Schemas`() throws {
    ConfiguredSchemas.cache.removeAll()
    expectNoDifference(ConfiguredSchemas.cache.count, 0)
    var parser = PartialsStream(initialValue: CachedRequest.Partial(), from: .json())
    try parser.next(#"{"text":"hello"}"#.utf8)
    expectNoDifference(CachedRequest(try parser.finish())?.text, "hello")
    expectNoDifference(ConfiguredSchemas.cache.contains(CachedRequest.Partial.self), true)

    var enumParser = PartialsStream(initialValue: CachedAction.Partial(), from: .json())
    try enumParser.next(#"{"text":{"body":"hi"}}"#.utf8)
    expectNoDifference(CachedAction(try enumParser.finish()), .text(body: "hi"))
    expectNoDifference(ConfiguredSchemas.cache.contains(CachedAction.Partial.self), true)
    expectNoDifference(
      ConfiguredSchemas.cache.contains(CachedAction.TextPayload.Partial.self),
      true
    )
  }

  @Test
  func `Incomplete Partials Have A Total Conversion`() {
    expectNoDifference(StreamSearchRequest(StreamSearchRequest.Partial()) == nil, true)
    let request = StreamSearchRequest(orInitial: StreamSearchRequest.Partial())
    expectNoDifference(request.query, "")
    expectNoDifference(request.scores, [])
    expectNoDifference(StreamAction(orInitial: StreamAction.Partial()), .idle(reason: nil))
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

// MARK: - Configured Streaming Models

@EdgeToolsGenerable(.additionalProperties(false), partialStrings: .string)
private struct StringStorageRequest {
  var title: String
  var tags: [String]
  var groups: [String: [String]]
  @EdgeToolsGuide(initialCapacity: 64, partialStrings: .streamString)
  var body: String
  var address: StreamAddress
}

@EdgeToolsGenerable
private struct MemberStringStorage {
  @EdgeToolsGuide(partialStrings: .string)
  var text: String
  var other: String
}

@EdgeToolsGenerable(keyDecodingStrategy: .convertFromSnakeCase)
private struct SnakeCaseRequest {
  var userID: Int
  var displayName: String
  @EdgeToolsGuide(key: "ExactName", .minLength(1))
  var exactName: String
}

private enum CustomKeys {
  static let strategy = StreamKeyDecodingStrategy.custom {
    "x_" + StreamKeyDecodingStrategy.convertFromSnakeCase.key(for: $0)
  }
}

@EdgeToolsGenerable(partialStrings: .string, keyDecodingStrategy: CustomKeys.strategy)
private enum CustomKeyAction: Equatable {
  @StreamParseableDefault
  case idle(reason: String?)
  case sendMessage(bodyText: String, String)
  case underscored(_bodyText: String)
}

@EdgeToolsGenerable(keyDecodingStrategy: CustomKeys.strategy)
private struct CustomKeyRequest {
  var userID: Int
  @EdgeToolsGuide(key: "fixed")
  var displayName: String
}

private enum ConfiguredSchemas {
  static let cache = StreamSchemaCache()
}

@EdgeToolsGenerable(schemaCache: ConfiguredSchemas.cache)
private struct CachedRequest {
  var text: String
}

@EdgeToolsGenerable(schemaCache: ConfiguredSchemas.cache)
private enum CachedAction: Equatable {
  @StreamParseableDefault
  case idle(reason: String?)
  case text(body: String)
}
