import CustomDump
import EdgeTools
import Testing

@Suite
struct `EdgeToolsGenericGenerable tests` {
  @Test
  func `Generic Collections Stream And Round Trip Their Partials`() throws {
    var stream = PartialsStream(initialValue: GenericPage<GenericItem>.Partial(), from: .json())
    try stream.next(#"{"items":[{"name":"he"#.utf8)
    let early = stream.current
    expectNoDifference(early.items?[0].name.map(String.init), "he")
    expectNoDifference(early.edgeToolsValue, ["items": [["name": "he"]]])
    let restored = try GenericPage<GenericItem>.Partial(edgeToolsValue: early.edgeToolsValue)
    expectNoDifference(restored.edgeToolsValue, early.edgeToolsValue)
    try stream.next(
      #"llo"}],"by_name":{"first":{"name":"world"}},"featured":{"name":"chosen"}}"#.utf8
    )
    let partial = try stream.finish()
    let page = try #require(GenericPage<GenericItem>(partial))
    expectNoDifference(page.items.map(\.name), ["hello"])
    expectNoDifference(page.byName["first"]?.name, "world")
    expectNoDifference(page.featured?.name, "chosen")
    expectNoDifference(page.edgeToolsValue, partial.edgeToolsValue)
    expectNoDifference(page.streamPartialValue.edgeToolsValue, page.edgeToolsValue)
    expectNoDifference(
      try GenericPage<GenericItem>(edgeToolsValue: page.edgeToolsValue).items.map(\.name),
      ["hello"]
    )
    expectNoDifference(early.items?[0].name.map(String.init), "he")
  }

  @Test
  func `Generic Scalar Schemas Stay Separate In The Configured Cache`() throws {
    var integers = PartialsStream(initialValue: GenericBox<Int>.Partial(), from: .json())
    var flags = PartialsStream(initialValue: GenericBox<Bool>.Partial(), from: .json())
    try integers.next(#"{"value":42}"#.utf8)
    try flags.next(#"{"value":true}"#.utf8)
    expectNoDifference(GenericBox<Int>(try integers.finish())?.value, 42)
    expectNoDifference(GenericBox<Bool>(try flags.finish())?.value, true)
    expectNoDifference(GenericSchemas.cache.contains(GenericBox<Int>.Partial.self), true)
    expectNoDifference(GenericSchemas.cache.contains(GenericBox<Bool>.Partial.self), true)
    let fields = try _edgeToolsRequireObjectValue(
      #require(GenericBox<Int>.edgeToolsGenerationSchema.objectValue?[.properties])
    )
    expectNoDifference(fields["value"], Int.edgeToolsGenerationSchema.edgeToolsValue)
  }

  @Test
  func `Generable Parameters Retain Value Conversion Without Stream Parsing`() throws {
    let box = try ValueOnlyBox<ValueOnly>(edgeToolsValue: ["value": "hello"])
    expectNoDifference(box.value.text, "hello")
    expectNoDifference(box.edgeToolsValue, ["value": "hello"])
  }

  @Test
  func `Generic Parameters Can Represent Explicit Null`() throws {
    var stream = PartialsStream(initialValue: GenericBox<Int?>.Partial(), from: .json())
    try stream.next(#"{"value":null}"#.utf8)
    let partial = try stream.finish()
    let box = try #require(GenericBox<Int?>(partial))
    expectNoDifference(box.value, nil)
    expectNoDifference(partial.edgeToolsValue, ["value": .null])
    expectNoDifference(GenericBox<Int?>(GenericBox<Int?>.Partial()) == nil, true)
    let restored = try GenericBox<Int?>.Partial(edgeToolsValue: partial.edgeToolsValue)
    expectNoDifference(GenericBox<Int?>(restored) == nil, false)
  }

  @Test
  func `Nested Generic Contexts And Phantom Parameters Support Streaming`() throws {
    var nested = PartialsStream(initialValue: GenericOuter<String>.Inner.Partial(), from: .json())
    try nested.next(#"{"value":"hello"}"#.utf8)
    expectNoDifference(GenericOuter<String>.Inner(try nested.finish())?.value, "hello")
    var phantom = PartialsStream(initialValue: GenericPhantom<PhantomTag>.Partial(), from: .json())
    try phantom.next(#"{"id":7}"#.utf8)
    expectNoDifference(GenericPhantom<PhantomTag>(try phantom.finish())?.id, 7)
  }

  @Test
  func `Associated Type Members Support Streaming`() throws {
    var stream = PartialsStream(
      initialValue: AssociatedRequest<IntStorage>.Partial(),
      from: .json()
    )
    try stream.next(#"{"value":42}"#.utf8)
    let partial = try stream.finish()
    expectNoDifference(AssociatedRequest<IntStorage>(partial)?.value, 42)
    expectNoDifference(partial.edgeToolsValue, ["value": 42])
    let restored = try AssociatedRequest<IntStorage>.Partial(edgeToolsValue: partial.edgeToolsValue)
    expectNoDifference(AssociatedRequest<IntStorage>(restored)?.value, 42)
  }

  @Test
  func `Generic Completed Conversions Retain Their Source Representation`() throws {
    var stream = PartialsStream(initialValue: GenericConvertedRequest<Int>.Partial(), from: .json())
    try stream.next(#"{"values":[1,"#.utf8)
    expectNoDifference(stream.current.values?.value?.values, nil)
    expectNoDifference(stream.current.values?.source.map { $0 }, [1])
    try stream.next("2]}".utf8)
    let partial = try stream.finish()
    let value = try #require(GenericConvertedRequest<Int>(partial))
    expectNoDifference(value.values.values, [1, 2])
    expectNoDifference(value.edgeToolsValue, ["values": [1, 2]])
    expectNoDifference(value.streamPartialValue.edgeToolsValue, partial.edgeToolsValue)
    let restored = try GenericConvertedRequest<Int>.Partial(edgeToolsValue: partial.edgeToolsValue)
    expectNoDifference(GenericConvertedRequest<Int>(restored)?.values.values, [1, 2])
  }

  @Test
  func `Ignored Generic Members Require No Parsing Conformance`() throws {
    var stream = PartialsStream(initialValue: IgnoredRequest<PhantomTag>.Partial(), from: .json())
    try stream.next(#"{"id":7,"scratch":"ignored"}"#.utf8)
    let value = try #require(IgnoredRequest<PhantomTag>(try stream.finish()))
    expectNoDifference(value.id, 7)
    expectNoDifference(value.scratch == nil, true)
    expectNoDifference(value.edgeToolsValue, ["id": 7])
  }

  @Test
  func `Generic Enums Retain Their Value Only Conformance`() throws {
    let action = try GenericAction<ValueOnly>(edgeToolsValue: ["value": ["_0": "hello"]])
    expectNoDifference(action.edgeToolsValue, ["value": ["_0": "hello"]])
  }

  @Test
  func `Conditional Sendable Partials Cross A Task Boundary`() async throws {
    let bytes = Array(#"{"value":"hello"}"#.utf8)
    let partial =
      try await Task.detached {
        var stream = PartialsStream(initialValue: GenericBox<String>.Partial(), from: .json())
        try stream.next(bytes)
        return try stream.finish()
      }
      .value
    expectNoDifference(GenericBox<String>(partial)?.value, "hello")
  }
}

// MARK: - Models

public enum GenericSchemas {
  public static let cache = StreamSchemaCache()
}

@EdgeToolsGenerable(schemaCache: GenericSchemas.cache)
public struct GenericBox<Value: EdgeToolsGenerable & StreamParseable & SendableMetatype>
where Value.Partial: EdgeToolsGenerable {
  public var value: Value
}

extension GenericBox.Partial: Sendable
where Value: StreamParseable, Value.Partial: EdgeToolsGenerable & Sendable {}

@EdgeToolsGenerable(keyDecodingStrategy: .convertFromSnakeCase)
private struct GenericPage<Item: EdgeToolsGenerable & StreamParseable & SendableMetatype>
where Item.Partial: EdgeToolsGenerable {
  var items: [Item]
  var byName: [String: Item]
  var featured: Item?
}

@EdgeToolsGenerable
private struct GenericItem {
  var name: String
}

struct GenericOuter<Value: EdgeToolsGenerable & StreamParseable & SendableMetatype>
where Value.Partial: EdgeToolsGenerable {
  @EdgeToolsGenerable
  struct Inner {
    var value: Value
  }
}

@EdgeToolsGenerable
private struct GenericPhantom<Tag> {
  var id: Int
}

private enum PhantomTag {}

private protocol GenericStorage: SendableMetatype {
  associatedtype Element: EdgeToolsGenerable & StreamParseable & SendableMetatype
  where Element.Partial: EdgeToolsGenerable
}

private enum IntStorage: GenericStorage {
  typealias Element = Int
}

@EdgeToolsGenerable
private struct AssociatedRequest<Storage: GenericStorage> {
  var value: Storage.Element
}

@EdgeToolsGenerable
private struct IgnoredRequest<Cache> {
  var id: Int
  @EdgeToolsIgnored
  var scratch: Cache?
}

@EdgeToolsGenerable
private enum GenericAction<Value: EdgeToolsGenerable> {
  @StreamParseableDefault
  case value(Value)
}

@EdgeToolsGenerable
private struct GenericConvertedRequest<Element: EdgeToolsGenerable & StreamParseableRoot & Sendable>
{
  @EdgeToolsGuide(completedConversion: GenericArrayConversion<Element>.self)
  var values: ConvertedElements<Element> = ConvertedElements(values: [])
}

private struct ConvertedElements<Element: Sendable>: Sendable {
  var values: [Element]
}

private enum GenericArrayConversion<Element: EdgeToolsGenerable & StreamParseableRoot & Sendable>:
  StreamCompletedValueConversion
{
  typealias Source = StreamArray<Element>

  static func convertToValue(_ source: borrowing Source.View) -> ConvertedElements<Element> {
    ConvertedElements(values: Array(source.value))
  }

  static func convertFromValue(_ value: ConvertedElements<Element>) -> Source {
    StreamArray(value.values)
  }
}

private struct ValueOnly: EdgeToolsGenerable {
  var text: String

  static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema { .string }

  init(edgeToolsValue: EdgeToolsValue) throws {
    self.text = try String(edgeToolsValue: edgeToolsValue)
  }

  var edgeToolsValue: EdgeToolsValue { .string(self.text) }
}

@EdgeToolsGenerable
private struct ValueOnlyBox<Value: EdgeToolsGenerable> {
  typealias Partial = Value
  var value: Value
}
