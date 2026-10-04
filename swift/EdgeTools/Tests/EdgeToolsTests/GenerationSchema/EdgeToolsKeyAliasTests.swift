import CustomDump
import EdgeTools
import Testing

@Suite
struct `EdgeToolsKeyAlias tests` {
  @Test
  func `Schemas Describe Canonical Keys Only`() throws {
    let schema = AliasedRequest.edgeToolsGenerationSchema
    let fields = try _edgeToolsRequireObjectValue(#require(schema.objectValue?[.properties]))
    expectNoDifference(Array(fields.keys), ["currentName", "currentAge", "fixed"])
    expectNoDifference(schema.objectValue?[.required], ["currentName"])
    expectNoDifference(
      fields["currentName"],
      EdgeToolsGenerationSchema(String.edgeToolsGenerationSchema, .minLength(1)).edgeToolsValue
    )
    let partialFields = try _edgeToolsRequireObjectValue(
      #require(AliasedRequest.Partial.edgeToolsGenerationSchema.objectValue?[.properties])
    )
    expectNoDifference(Array(partialFields.keys), ["currentName", "currentAge", "fixed"])
  }

  @Test
  func `Value And Partial Decoding Accept Aliases And Encode Canonical Keys`() throws {
    for key in ["currentName", "legacy_name"] {
      let input: EdgeToolsValue = .object([key: "hello", "old_age": 7, "fixed": "note"])
      let expected: EdgeToolsValue = ["currentName": "hello", "currentAge": 7, "fixed": "note"]
      expectNoDifference(try AliasedRequest(edgeToolsValue: input).edgeToolsValue, expected)
      let partial = try AliasedRequest.Partial(edgeToolsValue: input)
      expectNoDifference(partial.edgeToolsValue, expected)
      expectNoDifference(AliasedRequest(partial)?.edgeToolsValue, expected)
      var stream = PartialsStream(initialValue: AliasedRequest.Partial(), from: .json())
      try stream.next(input.orderedJSONString().utf8)
      expectNoDifference(AliasedRequest(try stream.finish())?.edgeToolsValue, expected)
    }
  }

  @Test
  func `Value Decoding Selects The Last Matching Entry Including Null`() throws {
    let inputs: [EdgeToolsValue] = [
      ["currentName": "first", "legacy_name": "last", "currentAge": 7, "old_age": .null],
      ["legacy_name": "first", "currentName": "last", "old_age": .null, "currentAge": 9]
    ]
    let expected: [EdgeToolsValue] = [
      ["currentName": "last"], ["currentName": "last", "currentAge": 9]
    ]
    for (input, expected) in zip(inputs, expected) {
      expectNoDifference(try AliasedRequest(edgeToolsValue: input).edgeToolsValue, expected)
      expectNoDifference(try AliasedRequest.Partial(edgeToolsValue: input).edgeToolsValue, expected)
    }
    #expect(throws: EdgeToolsValueTypeError.self) {
      try AliasedRequest(edgeToolsValue: ["legacy_name": "hello", "currentName": .null])
    }
  }

  @Test
  func `Incomplete Alias Snapshots Restore Under Canonical Keys`() throws {
    var stream = PartialsStream(initialValue: AliasedRequest.Partial(), from: .json())
    try stream.next(#"{"legacy_name":"hé"#.utf8)
    let snapshot = stream.current.edgeToolsValue
    expectNoDifference(snapshot, ["currentName": "hé"])
    expectNoDifference(try AliasedRequest.Partial(edgeToolsValue: snapshot).edgeToolsValue, snapshot)
    try stream.next(#"llo","old_age":7}"#.utf8)
    expectNoDifference(
      AliasedRequest(try stream.finish())?.edgeToolsValue,
      ["currentName": "héllo", "currentAge": 7]
    )
  }

  @Test
  func `Generic Aliases Preserve Explicit Null And Missing Fields`() throws {
    let missing = try AliasedBox<Int?>.Partial(edgeToolsValue: [:])
    expectNoDifference(AliasedBox<Int?>(missing) == nil, true)
    let input: EdgeToolsValue = ["value": 7, "old_value": .null]
    let restored = try AliasedBox<Int?>.Partial(edgeToolsValue: input)
    expectNoDifference(restored.edgeToolsValue, ["value": .null])
    expectNoDifference(try #require(AliasedBox<Int?>(restored)).value, nil)
    expectNoDifference(try AliasedBox<Int?>(edgeToolsValue: input).value, nil)
    var stream = PartialsStream(initialValue: AliasedBox<Int?>.Partial(), from: .json())
    try stream.next(input.orderedJSONString().utf8)
    expectNoDifference(try stream.finish().edgeToolsValue, ["value": .null])
  }

  @Test
  func `Converted Aliases Preserve Source And Work At Every Chunk Boundary`() throws {
    let input: EdgeToolsValue = ["old_token": "Héllo"]
    let decoded = try AliasedConvertedRequest(edgeToolsValue: input)
    expectNoDifference(decoded.token.text, "HÉLLO")
    expectNoDifference(decoded.edgeToolsValue, ["token": "héllo"])
    let restored = try AliasedConvertedRequest.Partial(edgeToolsValue: input)
    expectNoDifference(restored.edgeToolsValue, ["token": "Héllo"])
    expectNoDifference(restored.token?.value?.text, "HÉLLO")
    let bytes = Array(input.orderedJSONString().utf8)
    for split in 0...bytes.count {
      var stream = PartialsStream(initialValue: AliasedConvertedRequest.Partial(), from: .json())
      try stream.next(bytes[..<split])
      try stream.next(bytes[split...])
      let partial = try stream.finish()
      expectNoDifference(partial.edgeToolsValue, ["token": "Héllo"])
      expectNoDifference(AliasedConvertedRequest(partial)?.token.text, "HÉLLO")
    }
  }
}

// MARK: - Fixtures

private enum AliasKeys {
  static let strategy = StreamKeyDecodingStrategy.custom { "prefix_" + $0 }
}

@EdgeToolsGenerable(.additionalProperties(false), keyDecodingStrategy: AliasKeys.strategy)
private struct AliasedRequest {
  @EdgeToolsGuide(key: nil, keyNames: ["currentName", "legacy_name"], .minLength(1))
  var name: String
  @EdgeToolsGuide(keyNames: ["currentAge", "old_age"])
  var age: Int?
  @EdgeToolsGuide(key: "fixed", keyNames: nil)
  var note: String?
}

@EdgeToolsGenerable
private struct AliasedBox<Value: EdgeToolsGenerable & StreamParseable & SendableMetatype>
where Value.Partial: EdgeToolsGenerable {
  @EdgeToolsGuide(keyNames: ["value", "old_value"])
  var value: Value
}

@EdgeToolsGenerable
private struct AliasedConvertedRequest {
  @EdgeToolsGuide(keyNames: ["token", "old_token"], completedConversion: AliasConversion.self)
  var token: AliasToken = AliasToken(text: "")
}

private struct AliasToken: Sendable {
  var text: String
}

private enum AliasConversion: StreamCompletedValueConversion {
  typealias Source = StreamString

  static func convertToValue(_ source: borrowing Source.View) -> AliasToken {
    AliasToken(text: String(source.value).uppercased())
  }

  static func convertFromValue(_ value: AliasToken) -> Source {
    StreamString(value.text.lowercased())
  }
}
