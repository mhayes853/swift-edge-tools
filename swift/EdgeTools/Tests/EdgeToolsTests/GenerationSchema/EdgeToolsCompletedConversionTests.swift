import CustomDump
import EdgeTools
import Testing

@Suite
struct `EdgeToolsCompletedConversion tests` {
  @Test
  func `Conversions Run Once And Reverse Conversion Reuses The Cached Value`() throws {
    CountingConversion.calls.withLock { $0 = (0, 0) }
    var stream = PartialsStream(initialValue: CountingRequest.Partial(), from: .json())
    try stream.next(#"{"token":"hello"}"#.utf8)
    expectNoDifference(CountingConversion.calls.withLock { $0.to }, 1)
    let partial = try stream.finish()
    let first = try #require(CountingRequest(partial))
    let second = try #require(CountingRequest(partial))
    expectNoDifference(first.token, second.token)
    expectNoDifference(partial.edgeToolsValue, ["token": "hello"])
    expectNoDifference(CountingConversion.calls.withLock { $0.to }, 1)
    let reversed = first.streamPartialValue
    expectNoDifference(reversed.token?.value, first.token)
    expectNoDifference(CountingRequest(reversed)?.token, first.token)
    expectNoDifference(CountingConversion.calls.withLock { $0.to }, 1)
    expectNoDifference(CountingConversion.calls.withLock { $0.from }, 1)

    CountingConversion.calls.withLock { $0 = (0, 0) }
    let restored = try CountingRequest.Partial(edgeToolsValue: ["token": "HeLLo"])
    expectNoDifference(restored.edgeToolsValue, ["token": "HeLLo"])
    expectNoDifference(CountingRequest(restored)?.token.text, "HeLLo")
    expectNoDifference(CountingConversion.calls.withLock { $0.to }, 1)
    expectNoDifference(CountingConversion.calls.withLock { $0.from }, 0)
  }

  @Test
  func `Completed Conversions Work At Every Chunk Boundary`() throws {
    let bytes = Array(#"{"wire_token":"héllo","created_at":42,"pair":[3,4]}"#.utf8)
    for split in 0...bytes.count {
      var stream = PartialsStream(initialValue: ConvertedRequest.Partial(), from: .json())
      try stream.next(bytes[..<split])
      try stream.next(bytes[split...])
      let partial = try stream.finish()
      expectNoDifference(ConvertedRequest(partial)?.token, ConvertedToken(text: "HÉLLO"))
      expectNoDifference(ConvertedRequest(partial)?.createdAt, ConvertedTimestamp(seconds: 42))
      expectNoDifference(ConvertedRequest(partial)?.pair, ConvertedPair(first: 3, second: 4))
    }
  }

  @Test
  func `Schemas Describe The Source With Guides And Nullable Values`() throws {
    let schema = ConvertedRequest.edgeToolsGenerationSchema
    let properties = try _edgeToolsRequireObjectValue(#require(schema.objectValue?[.properties]))
    expectNoDifference(
      properties["wire_token"],
      EdgeToolsGenerationSchema(.string, .minLength(1)).edgeToolsValue
    )
    expectNoDifference(
      properties["created_at"],
      Int.edgeToolsGenerationSchema.nullable().edgeToolsValue
    )
    expectNoDifference(
      properties["pair"],
      StreamArray<Int>.edgeToolsGenerationSchema.edgeToolsValue
    )
    expectNoDifference(schema.objectValue?[.required], ["wire_token", "pair"])
    expectNoDifference(
      ConvertedRequest.Partial.edgeToolsGenerationSchema.objectValue?[.required],
      nil
    )
    let partialProperties = try _edgeToolsRequireObjectValue(
      #require(ConvertedRequest.Partial.edgeToolsGenerationSchema.objectValue?[.properties])
    )
    expectNoDifference(
      partialProperties["wire_token"],
      EdgeToolsGenerationSchema(.string.nullable(), .minLength(1)).edgeToolsValue
    )
  }

  @Test
  func `Value Conversion Uses The Source For Domain Types`() throws {
    let input: EdgeToolsValue = ["wire_token": "Hello", "created_at": 42, "pair": [3, 4]]
    let value = try ConvertedRequest(edgeToolsValue: input)
    expectNoDifference(value.token, ConvertedToken(text: "HELLO"))
    expectNoDifference(value.createdAt, ConvertedTimestamp(seconds: 42))
    expectNoDifference(value.pair, ConvertedPair(first: 3, second: 4))
    let canonical: EdgeToolsValue = ["wire_token": "hello", "created_at": 42, "pair": [3, 4]]
    expectNoDifference(value.edgeToolsValue, canonical)
    expectNoDifference(try ConvertedRequest(edgeToolsValue: canonical), value)
    expectNoDifference(ConvertedRequest(value.streamPartialValue), value)
  }

  @Test
  func `Converted Strings And Containers Wait For Completion`() throws {
    var stream = PartialsStream(initialValue: ConvertedRequest.Partial(), from: .json())
    try stream.next(#"{"wire_token":"He"#.utf8)
    let early = stream.current
    expectNoDifference(early.token.map { String($0.source) }, "He")
    expectNoDifference(early.token?.value, nil)
    expectNoDifference(early.edgeToolsValue, ["wire_token": "He"])

    try stream.next(#"llo","pair":[3,"#.utf8)
    expectNoDifference(stream.current.token?.value, ConvertedToken(text: "HELLO"))
    expectNoDifference(stream.current.pair?.source.map { $0 }, [3])
    expectNoDifference(stream.current.pair?.value, nil)
    expectNoDifference(ConvertedRequest(stream.current), nil)
    try stream.next("4]}".utf8)
    let partial = try stream.finish()
    expectNoDifference(ConvertedRequest(partial)?.pair, ConvertedPair(first: 3, second: 4))
    expectNoDifference(early.token.map { String($0.source) }, "He")
    expectNoDifference(early.token?.value, nil)
  }

  @Test
  func `Partial Value Restoration Preserves The Original Source`() throws {
    let input: EdgeToolsValue = ["wire_token": "Hello", "pair": [3, 4]]
    let partial = try ConvertedRequest.Partial(edgeToolsValue: input)
    expectNoDifference(partial.edgeToolsValue, input)
    expectNoDifference(partial.token?.value, ConvertedToken(text: "HELLO"))
    expectNoDifference(ConvertedRequest(partial)?.token, ConvertedToken(text: "HELLO"))
    expectNoDifference(try ConvertedRequest.Partial(edgeToolsValue: [:]).token == nil, true)
    expectNoDifference(
      ConvertedRequest(orInitial: ConvertedRequest.Partial()).token.text,
      "DEFAULT"
    )
    expectNoDifference(
      ConvertedRequest(orInitial: ConvertedRequest.Partial()).pair,
      ConvertedPair(first: 0, second: 0)
    )
  }

  @Test
  func `Optional Conversions Distinguish Missing Null And Incomplete Values`() throws {
    var stream = PartialsStream(initialValue: OptionalConvertedRequest.Partial(), from: .json())
    try stream.next(#"{"token":"he"#.utf8)
    expectNoDifference(OptionalConvertedRequest(stream.current), nil)
    try stream.next(#"llo","token":null}"#.utf8)
    expectNoDifference(OptionalConvertedRequest(try stream.finish())?.token, nil)
    expectNoDifference(try OptionalConvertedRequest(edgeToolsValue: [:]).token, nil)
    expectNoDifference(try OptionalConvertedRequest(edgeToolsValue: ["token": .null]).token, nil)
    expectNoDifference(
      try OptionalConvertedRequest.Partial(edgeToolsValue: ["token": .null]).token == nil,
      true
    )
    expectNoDifference(try OptionalConvertedRequest(edgeToolsValue: [:]).edgeToolsValue, [:])
  }

  @Test
  func `Conversion Failures Retain Their Error Types`() throws {
    #expect(throws: TokenConversionError.invalid) {
      try ConvertedRequest(edgeToolsValue: ["wire_token": "bad", "pair": [1, 2]])
    }
    #expect(throws: TokenConversionError.invalid) {
      try ConvertedRequest.Partial(edgeToolsValue: ["wire_token": "bad"])
    }
    #expect(throws: EdgeToolsValueTypeError.self) {
      try ConvertedRequest(edgeToolsValue: ["wire_token": 1, "pair": [1, 2]])
    }
    #expect(throws: EdgeToolsValueTypeError.self) {
      try ConvertedRequest.Partial(edgeToolsValue: ["wire_token": 1])
    }
    #expect(throws: TokenConversionError.invalid) {
      try ConvertedRequest(edgeToolsValue: ["wire_token": "hello", "pair": [1]])
    }
    var stream = PartialsStream(initialValue: ConvertedRequest.Partial(), from: .json())
    #expect(throws: JSONParsingError.self) {
      try stream.next(#"{"wire_token":"bad"}"#.utf8)
    }
    expectNoDifference(stream.current.token?.conversionError, .invalid)
    expectNoDifference(stream.current.token?.value, nil)
    expectNoDifference(stream.current.token.map { String($0.source) }, "bad")
  }

  @Test
  func `Object Sources Convert Only After Their Closing Brace`() throws {
    var stream = PartialsStream(initialValue: ObjectConvertedRequest.Partial(), from: .json())
    try stream.next(#"{"pair":{"first":3,"second":4"#.utf8)
    expectNoDifference(stream.current.pair?.value, nil)
    expectNoDifference(stream.current.pair?.source["first"], 3)
    try stream.next("}}".utf8)
    let partial = try stream.finish()
    expectNoDifference(ObjectConvertedRequest(partial)?.pair, ConvertedPair(first: 3, second: 4))
    let input: EdgeToolsValue = ["pair": ["first": 3, "second": 4]]
    expectNoDifference(try ObjectConvertedRequest(edgeToolsValue: input).edgeToolsValue, input)
    expectNoDifference(
      try ObjectConvertedRequest.Partial(edgeToolsValue: input).edgeToolsValue,
      input
    )
  }
}

// MARK: - Models

private struct ConvertedToken: Equatable, Sendable {
  var text: String
}

private struct ConvertedTimestamp: Equatable, Sendable {
  var seconds: Int
}

private struct ConvertedPair: Equatable, Sendable {
  var first: Int
  var second: Int
}

@EdgeToolsGenerable(partialStrings: .string, keyDecodingStrategy: .convertFromSnakeCase)
private struct ConvertedRequest: Equatable {
  @EdgeToolsGuide(key: "wire_token", completedConversion: TokenConversion.self, .minLength(1))
  var token: ConvertedToken = ConvertedToken(text: "DEFAULT")

  @EdgeToolsGuide(completedConversion: TimestampConversion.self)
  var createdAt: ConvertedTimestamp?

  @EdgeToolsGuide(completedConversion: PairConversion.self)
  var pair: ConvertedPair = ConvertedPair(first: 0, second: 0)
}

@EdgeToolsGenerable
private struct OptionalConvertedRequest: Equatable {
  @EdgeToolsGuide(completedConversion: TokenConversion.self)
  var token: ConvertedToken?
}

@EdgeToolsGenerable
private struct ObjectConvertedRequest {
  @EdgeToolsGuide(completedConversion: ObjectPairConversion.self)
  var pair: ConvertedPair = ConvertedPair(first: 0, second: 0)
}

@EdgeToolsGenerable
private struct CountingRequest {
  @EdgeToolsGuide(completedConversion: CountingConversion.self)
  var token: ConvertedToken = ConvertedToken(text: "DEFAULT")
}

// MARK: - Strategies

private enum CountingConversion: StreamCompletedValueConversion {
  typealias Source = StreamString
  static let calls = LockBox((to: 0, from: 0))

  static func convertToValue(_ source: borrowing Source.View) -> ConvertedToken {
    Self.calls.withLock { $0.to += 1 }
    return ConvertedToken(text: String(source.value))
  }

  static func convertFromValue(_ value: ConvertedToken) -> StreamString {
    Self.calls.withLock { $0.from += 1 }
    return StreamString(value.text)
  }
}

private enum TokenConversionError: Error {
  case invalid
}

private enum TokenConversion: StreamCompletedValueConversion {
  typealias Source = StreamString

  static func convertToValue(_ source: borrowing Source.View) throws(TokenConversionError)
    -> ConvertedToken
  {
    let text = String(source.value)
    guard text != "bad" else {
      throw .invalid
    }
    return ConvertedToken(text: text.uppercased())
  }

  static func convertFromValue(_ value: ConvertedToken) -> StreamString {
    StreamString(value.text.lowercased())
  }
}

private enum TimestampConversion: StreamCompletedValueConversion {
  typealias Source = Int

  static func convertToValue(_ source: borrowing Source.View) -> ConvertedTimestamp {
    ConvertedTimestamp(seconds: source.value)
  }

  static func convertFromValue(_ value: ConvertedTimestamp) -> Int {
    value.seconds
  }
}

private enum PairConversion: StreamCompletedValueConversion {
  typealias Source = StreamArray<Int>

  static func convertToValue(_ source: borrowing Source.View) throws(TokenConversionError)
    -> ConvertedPair
  {
    guard source.count == 2 else {
      throw .invalid
    }
    return ConvertedPair(first: source.value[0], second: source.value[1])
  }

  static func convertFromValue(_ value: ConvertedPair) -> Source {
    [value.first, value.second]
  }
}

private enum ObjectPairConversion: StreamCompletedValueConversion {
  typealias Source = StreamDictionary<Int>

  static func convertToValue(_ source: borrowing Source.View) throws(TokenConversionError)
    -> ConvertedPair
  {
    guard let first = source.value["first"], let second = source.value["second"] else {
      throw .invalid
    }
    return ConvertedPair(first: first, second: second)
  }

  static func convertFromValue(_ value: ConvertedPair) -> Source {
    ["first": value.first, "second": value.second]
  }
}
