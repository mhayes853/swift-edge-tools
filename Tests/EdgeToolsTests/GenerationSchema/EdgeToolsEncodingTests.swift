import CustomDump
import EdgeTools
import Foundation
import Testing

@Suite
struct `EdgeToolsEncoding tests` {
  @Test
  func `String Encodes To String Value`() {
    expectNoDifference("blob".edgeToolsValue, .string("blob"))
  }

  @Test
  func `Bool Encodes To Boolean Value`() {
    expectNoDifference(true.edgeToolsValue, .boolean(true))
    expectNoDifference(false.edgeToolsValue, .boolean(false))
  }

  @Test
  func `Signed Integers Encode To Integer Value`() {
    expectNoDifference((-1).edgeToolsValue, .integer(-1))
    expectNoDifference(Int8(2).edgeToolsValue, .integer(2))
    expectNoDifference(Int16(3).edgeToolsValue, .integer(3))
    expectNoDifference(Int32(4).edgeToolsValue, .integer(4))
    expectNoDifference(Int64(5).edgeToolsValue, .integer(5))
    expectNoDifference(Int64.max.edgeToolsValue, .integer(Int(Int64.max)))
  }

  @Test
  func `Unsigned Integers Encode To Integer Value`() {
    expectNoDifference(UInt(2).edgeToolsValue, .integer(2))
    expectNoDifference(UInt8(3).edgeToolsValue, .integer(3))
    expectNoDifference(UInt16(4).edgeToolsValue, .integer(4))
    expectNoDifference(UInt32(5).edgeToolsValue, .integer(5))
    expectNoDifference(UInt64(6).edgeToolsValue, .integer(6))
    expectNoDifference(UInt64(Int.max).edgeToolsValue, .integer(Int.max))
  }

  @Test
  func `Doubles And Floats Encode To Number Value`() {
    expectNoDifference((1.5).edgeToolsValue, .number(1.5))
    expectNoDifference(Float(2.5).edgeToolsValue, .number(2.5))
  }

  @Test
  func `Data Encodes To UTF8 String Value`() {
    let data = Data("blob".utf8)
    expectNoDifference(data.edgeToolsValue, .string("blob"))
  }

  @Test
  func `Decimal Encodes To String Value`() {
    expectNoDifference(Decimal(1.5).edgeToolsValue, .string("1.5"))
    expectNoDifference(Decimal(string: "12345.67890123456789")!.edgeToolsValue, .string("12345.67890123456789"))
  }

  #if canImport(CoreGraphics)
    @Test
    func `CGFloat Encodes To Number Value`() {
      expectNoDifference(CGFloat(2.5).edgeToolsValue, .number(2.5))
    }
  #endif

  @Test
  func `Optional Encodes To Null Or Inner Value`() {
    expectNoDifference(String?.none.edgeToolsValue, .null)
    expectNoDifference(String?.some("blob").edgeToolsValue, .string("blob"))
    expectNoDifference(Int?.some(42).edgeToolsValue, .integer(42))
  }

  @Test
  func `Array Encodes To Array Value`() {
    expectNoDifference([Int]().edgeToolsValue, .array([]))
    expectNoDifference([1, 2, 3].edgeToolsValue, .array([.integer(1), .integer(2), .integer(3)]))
    expectNoDifference(["a", "b"].edgeToolsValue, .array([.string("a"), .string("b")]))
  }

  @Test
  func `Dictionary Encodes To Object Value`() {
    expectNoDifference([String: Int]().edgeToolsValue, .object([:]))
    expectNoDifference(
      ["one": 1, "two": 2].edgeToolsValue,
      .object(["one": .integer(1), "two": .integer(2)])
    )
  }

  @Test
  func `EdgeToolsValue Encodes To Itself`() {
    expectNoDifference(EdgeToolsValue.null.edgeToolsValue, .null)
    expectNoDifference(EdgeToolsValue.string("blob").edgeToolsValue, .string("blob"))
  }

  @Test
  func `Macro Generated Type Round Trips Through EdgeToolsValue`() throws {
    let value: EdgeToolsValue = [
      "name": "Blob",
      "display_age": 42,
      "nickname": "blob",
      "tags": ["swift", "needle"],
      "metadata": ["role": "admin"],
      "address": ["city": "Brooklyn"]
    ]
    let user = try MacroEncodingUser(edgeToolsValue: value)

    let encoded = user.edgeToolsValue
    let decoded = try MacroEncodingUser(edgeToolsValue: encoded)
    expectNoDifference(decoded, user)
    expectNoDifference(decoded.ignoredDefault, "default")
  }

  @Test
  func `Macro Generated Encoding Honors Key Overrides`() throws {
    let value: EdgeToolsValue = [
      "name": "Blob",
      "display_age": 42,
      "tags": [],
      "metadata": [:],
      "address": ["city": "Brooklyn"]
    ]
    let user = try MacroEncodingUser(edgeToolsValue: value)

    guard case .object(let object) = user.edgeToolsValue else {
      Issue.record("Expected object value.")
      return
    }
    expectNoDifference(Array(object.keys), ["name", "display_age", "tags", "metadata", "address"])
    expectNoDifference(object["display_age"], .integer(42))
  }

  @Test
  func `Macro Generated Encoding Omits Nil Optionals And Ignored Properties`() throws {
    let value: EdgeToolsValue = [
      "name": "Blob",
      "display_age": 1,
      "tags": [],
      "metadata": [:],
      "address": ["city": "X"]
    ]
    let user = try MacroEncodingUser(edgeToolsValue: value)

    guard case .object(let object) = user.edgeToolsValue else {
      Issue.record("Expected object value.")
      return
    }
    expectNoDifference(object["nickname"], nil)
    expectNoDifference(object["tags"], .array([]))
    expectNoDifference(object["metadata"], .object([:]))
    expectNoDifference(object["ignoredOptional"], nil)
    expectNoDifference(object["ignoredDefault"], nil)
  }

  @Test
  func `Optional Nil Round Trips Through EdgeToolsValue`() throws {
    let value: EdgeToolsValue = [
      "name": "Blob",
      "display_age": 1,
      "tags": [],
      "metadata": [:],
      "address": ["city": "X"]
    ]
    let user = try MacroEncodingUser(edgeToolsValue: value)

    let decoded = try MacroEncodingUser(edgeToolsValue: user.edgeToolsValue)
    expectNoDifference(decoded.nickname, nil)
    expectNoDifference(decoded.ignoredOptional, nil)
    expectNoDifference(decoded.ignoredDefault, "default")
  }
}

@EdgeToolsGenerable
private struct MacroEncodingUser: Equatable {
  var name: String
  @EdgeToolsGuide(key: "display_age")
  var displayAge: Int
  var nickname: String?
  var tags: [String]
  var metadata: [String: String]
  var address: MacroEncodingAddress
  @EdgeToolsIgnored
  var ignoredOptional: String?
  @EdgeToolsIgnored
  var ignoredDefault: String = "default"
}

@EdgeToolsGenerable
private struct MacroEncodingAddress: Equatable {
  var city: String
}