import CustomDump
import EdgeTools
import Testing

@Suite
struct `EdgeToolsBase64Generable tests` {
  @Test
  func `Base64FieldsAcceptAliasesAndEncodeCanonicalKeys`() throws {
    let vectors: [(String, [UInt8])] = [
      ("", []), ("Zg==", [102]), ("Zm8=", [102, 111]), ("Zm9v", [102, 111, 111]),
      ("AAEC/f7/", [0, 1, 2, 253, 254, 255])
    ]
    for (text, bytes) in vectors {
      let input: EdgeToolsValue = ["payload": .string(text)]
      let expected: EdgeToolsValue = ["contents": .string(text)]
      let value = try BinaryRequest(edgeToolsValue: input)
      expectNoDifference(value.contents.bytes, bytes)
      expectNoDifference(value.edgeToolsValue, expected)
      let partial = try BinaryRequest.Partial(edgeToolsValue: input)
      expectNoDifference(partial.edgeToolsValue, expected)
      expectNoDifference(BinaryRequest(streamPartial: partial)?.contents.bytes, bytes)
    }
  }

  @Test
  func `BinaryFieldsStreamAtEveryChunkBoundary`() throws {
    let text = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0+Pw=="
    let bytes = Array((EdgeToolsValue.object(["payload": .string(text)])).orderedJSONString().utf8)
    for split in 0...bytes.count {
      var stream = PartialsStream<BinaryRequest>(from: .json())
      try stream.next(bytes[..<split])
      try stream.next(bytes[split...])
      let value = try stream.finish()
      expectNoDifference(value.contents.bytes, Array(UInt8(0)..<64))
      expectNoDifference(value.edgeToolsValue, ["contents": .string(text)])
    }
    var escaped = PartialsStream<BinaryRequest>(from: .json())
    try escaped.next(#"{"contents":"\u002f\/8="}"#.utf8)
    let escapedValue = try escaped.finish()
    expectNoDifference(escapedValue.contents.bytes, [255, 255])
  }

  @Test
  func `BinarySnapshotsRetainDecodedGroupsWhileTheStringIsIncomplete`() throws {
    var stream = PartialsStream<BinaryRequest>(from: .json())
    try stream.next(#"{"contents":"Zm9vZ"#.utf8)
    let early = stream.current
    expectNoDifference(early.contents?.bytes, [102, 111, 111])
    expectNoDifference(BinaryRequest(streamPartial: early) == nil, true)
    expectNoDifference(early.edgeToolsValue, ["contents": "Zm9v"])
    let restored = try BinaryRequest.Partial(edgeToolsValue: early.edgeToolsValue)
    expectNoDifference(BinaryRequest(streamPartial: restored)?.contents.bytes, [102, 111, 111])
    try stream.next(#"g=="}"#.utf8)
    let value = try stream.finish()
    expectNoDifference(value.contents.bytes, [102, 111, 111, 102])
    expectNoDifference(early.contents?.bytes, [102, 111, 111])
  }

  @Test
  func `Base64ValuesRejectInvalidCharactersAndUnfinishedGroups`() {
    #expect(throws: EdgeToolsValueTypeError.self) {
      try BinaryRequest(edgeToolsValue: ["contents": 1])
    }
    for text in ["Zm 9v", "____", "Zg===", "Zg==AAAA"] {
      #expect(throws: StreamSinkFailure(reason: .conversionFailed)) {
        try BinaryRequest(edgeToolsValue: ["contents": .string(text)])
      }
      #expect(throws: JSONParsingError.self) {
        var stream = PartialsStream<BinaryRequest>(from: .json())
        try stream.next((EdgeToolsValue.object(["contents": .string(text)])).orderedJSONString().utf8)
        return try stream.finish()
      }
    }
    for text in ["Z", "Zg", "Zg="] {
      #expect(throws: StreamParsingError.incompleteValue) {
        try BinaryRequest.Partial(edgeToolsValue: ["contents": .string(text)])
      }
      #expect(throws: StreamParsingError.incompleteValue) {
        var stream = PartialsStream<BinaryRequest>(from: .json())
        try stream.next((EdgeToolsValue.object(["contents": .string(text)])).orderedJSONString().utf8)
        return try stream.finish()
      }
    }
  }

  @Test
  func `CompletedConversionsUseBase64SourcesForSchemasValuesAndStreaming`() throws {
    let input: EdgeToolsValue = ["bytes": "AAEC/f7/"]
    let expected: [UInt8] = [0, 1, 2, 253, 254, 255]
    expectNoDifference(
      ConvertedBinaryRequest.edgeToolsGenerationSchema.objectValue?[.properties],
      ["bytes": StreamBase64Data.edgeToolsGenerationSchema.edgeToolsValue]
    )
    expectNoDifference(try ConvertedBinaryRequest(edgeToolsValue: input).bytes, expected)
    let partial = try ConvertedBinaryRequest.Partial(edgeToolsValue: input)
    expectNoDifference(partial.edgeToolsValue, input)
    expectNoDifference(ConvertedBinaryRequest(streamPartial: partial)?.bytes, expected)
    var stream = PartialsStream<ConvertedBinaryRequest>(from: .json())
    try stream.next(input.orderedJSONString().utf8)
    let value = try stream.finish()
    expectNoDifference(value.bytes, expected)
    expectNoDifference(value.edgeToolsValue, input)
    #expect(throws: JSONParsingError.self) {
      var incomplete = PartialsStream<ConvertedBinaryRequest>(from: .json())
      try incomplete.next(#"{"bytes":"Zg"}"#.utf8)
      return try incomplete.finish()
    }
  }
}

// MARK: - Fixtures

@EdgeToolsGenerable
private struct BinaryRequest {
  @EdgeToolsGuide(keyNames: ["contents", "payload"])
  var contents: StreamBase64Data
}

@EdgeToolsGenerable
private struct ConvertedBinaryRequest {
  @EdgeToolsGuide(completedConversion: BinaryConversion.self)
  var bytes: [UInt8] = []
}

private enum BinaryConversion: StreamCompletedValueConversion {
  typealias Source = StreamBase64Data

  static func convertToValue(_ source: borrowing Source.View) throws(StreamParsingError) -> [UInt8] {
    guard let value = Source(streamPartial: source.value) else {
      throw .incompleteValue
    }
    return value.bytes
  }

  static func convertFromValue(_ value: [UInt8]) -> Source {
    Source(value)
  }
}
