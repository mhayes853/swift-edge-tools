import CustomDump
import Foundation
import EdgeTools
import Testing

@Suite
struct `EdgeToolsGenerable initialization tests` {
  @Test
  func `Initializes String`() throws {
    expectNoDifference(try String(edgeToolsValue: "blob"), "blob")
  }

  @Test
  func `Initializes Bool`() throws {
    expectNoDifference(try Bool(edgeToolsValue: true), true)
  }

  @Test
  func `Initializes Double`() throws {
    expectNoDifference(try Double(edgeToolsValue: 1.5), 1.5)
    expectNoDifference(try Double(edgeToolsValue: 1), 1)
  }

  @Test
  func `Initializes Float`() throws {
    expectNoDifference(try Float(edgeToolsValue: 1.5), 1.5)
    expectNoDifference(try Float(edgeToolsValue: 1), 1)
  }

  @Test
  func `Initializes Int8`() throws {
    expectNoDifference(try Int8(edgeToolsValue: 1), 1)
  }

  @Test
  func `Initializes Int16`() throws {
    expectNoDifference(try Int16(edgeToolsValue: 1), 1)
  }

  @Test
  func `Initializes Int32`() throws {
    expectNoDifference(try Int32(edgeToolsValue: 1), 1)
  }

  @Test
  func `Initializes Int64`() throws {
    expectNoDifference(try Int64(edgeToolsValue: 1), 1)
  }

  @Test
  func `Initializes Int`() throws {
    expectNoDifference(try Int(edgeToolsValue: 1), 1)
  }

  @Test
  func `Initializes UInt8`() throws {
    expectNoDifference(try UInt8(edgeToolsValue: 1), 1)
  }

  @Test
  func `Initializes UInt16`() throws {
    expectNoDifference(try UInt16(edgeToolsValue: 1), 1)
  }

  @Test
  func `Initializes UInt32`() throws {
    expectNoDifference(try UInt32(edgeToolsValue: 1), 1)
  }

  @Test
  func `Initializes UInt64`() throws {
    expectNoDifference(try UInt64(edgeToolsValue: 1), 1)
  }

  @Test
  func `Initializes UInt`() throws {
    expectNoDifference(try UInt(edgeToolsValue: 1), 1)
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  @Test
  func `Initializes Int128`() throws {
    expectNoDifference(try Int128(edgeToolsValue: 1), 1)
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  @Test
  func `Initializes UInt128`() throws {
    expectNoDifference(try UInt128(edgeToolsValue: 1), 1)
  }

  @Test
  func `Initializes Data From UTF8 String`() throws {
    expectNoDifference(try Data(edgeToolsValue: "blob"), Data("blob".utf8))
  }

  @Test
  func `Initializes Decimal`() throws {
    expectNoDifference(try Decimal(edgeToolsValue: 1.5), Decimal(1.5))
    expectNoDifference(try Decimal(edgeToolsValue: 1), Decimal(1))
  }

  @Test
  func `Initializes Optional`() throws {
    expectNoDifference(try String?(edgeToolsValue: .null), nil)
    expectNoDifference(try String?(edgeToolsValue: "blob"), "blob")
  }

  @Test
  func `Initializes Array`() throws {
    expectNoDifference(try [Int](edgeToolsValue: [1, 2, 3]), [1, 2, 3])
  }

  @Test
  func `Initializes Dictionary`() throws {
    expectNoDifference(try [String: Int](edgeToolsValue: ["one": 1, "two": 2]), ["one": 1, "two": 2])
  }

  @Test
  func `Throws Type Error For Invalid Primitive Value`() {
    #expect(throws: EdgeToolsValueTypeError(expected: .string, received: .integer)) {
      try String(edgeToolsValue: 1)
    }
    #expect(throws: EdgeToolsValueTypeError(expected: .boolean, received: .string)) {
      try Bool(edgeToolsValue: "blob")
    }
    #expect(throws: EdgeToolsValueTypeError(expected: .number, received: .string)) {
      try Double(edgeToolsValue: "blob")
    }
    #expect(throws: EdgeToolsValueTypeError(expected: .integer, received: .number)) {
      try Int(edgeToolsValue: 1.5)
    }
    #expect(throws: EdgeToolsValueTypeError(expected: .array, received: .object)) {
      try [String](edgeToolsValue: [:])
    }
    #expect(throws: EdgeToolsValueTypeError(expected: .object, received: .array)) {
      try [String: String](edgeToolsValue: [])
    }
  }

  @Test
  func `Throws Type Error For Integer Overflow And Underflow`() {
    #expect(throws: EdgeToolsValueTypeError(expected: .integer, received: .integer)) {
      try Int8(edgeToolsValue: 999)
    }
    #expect(throws: EdgeToolsValueTypeError(expected: .integer, received: .integer)) {
      try UInt(edgeToolsValue: -1)
    }
  }

  @Test
  func `Initializes Macro Generated Type`() throws {
    let value: EdgeToolsValue = [
      "name": "Blob",
      "display_age": 42,
      "nickname": .null,
      "tags": ["swift", "needle"],
      "metadata": ["role": "admin"],
      "address": ["city": "Brooklyn"]
    ]

    let user = try MacroUser(edgeToolsValue: value)

    expectNoDifference(user.name, "Blob")
    expectNoDifference(user.age, 42)
    expectNoDifference(user.nickname, nil)
    expectNoDifference(user.tags, ["swift", "needle"])
    expectNoDifference(user.metadata, ["role": "admin"])
    expectNoDifference(user.address.city, "Brooklyn")
    expectNoDifference(user.ignoredOptional, nil)
    expectNoDifference(user.ignoredDefault, "default")
  }
}

@EdgeToolsGenerable
private struct MacroUser: Equatable {
  var name: String
  @EdgeToolsGuide(key: "display_age")
  var age: Int
  var nickname: String?
  var tags: [String]
  var metadata: [String: String]
  var address: Address
  @EdgeToolsIgnored
  var ignoredOptional: String?
  @EdgeToolsIgnored
  var ignoredDefault: String = "default"
}

@EdgeToolsGenerable
private struct Address: Equatable {
  var city: String
}
