import CustomDump
import Foundation
import Needle
import Testing

@Suite
struct `NeedleGenerable initialization tests` {
  @Test
  func `Initializes String`() throws {
    expectNoDifference(try String(needleValue: "blob"), "blob")
  }

  @Test
  func `Initializes Bool`() throws {
    expectNoDifference(try Bool(needleValue: true), true)
  }

  @Test
  func `Initializes Double`() throws {
    expectNoDifference(try Double(needleValue: 1.5), 1.5)
    expectNoDifference(try Double(needleValue: 1), 1)
  }

  @Test
  func `Initializes Float`() throws {
    expectNoDifference(try Float(needleValue: 1.5), 1.5)
    expectNoDifference(try Float(needleValue: 1), 1)
  }

  @Test
  func `Initializes Int8`() throws {
    expectNoDifference(try Int8(needleValue: 1), 1)
  }

  @Test
  func `Initializes Int16`() throws {
    expectNoDifference(try Int16(needleValue: 1), 1)
  }

  @Test
  func `Initializes Int32`() throws {
    expectNoDifference(try Int32(needleValue: 1), 1)
  }

  @Test
  func `Initializes Int64`() throws {
    expectNoDifference(try Int64(needleValue: 1), 1)
  }

  @Test
  func `Initializes Int`() throws {
    expectNoDifference(try Int(needleValue: 1), 1)
  }

  @Test
  func `Initializes UInt8`() throws {
    expectNoDifference(try UInt8(needleValue: 1), 1)
  }

  @Test
  func `Initializes UInt16`() throws {
    expectNoDifference(try UInt16(needleValue: 1), 1)
  }

  @Test
  func `Initializes UInt32`() throws {
    expectNoDifference(try UInt32(needleValue: 1), 1)
  }

  @Test
  func `Initializes UInt64`() throws {
    expectNoDifference(try UInt64(needleValue: 1), 1)
  }

  @Test
  func `Initializes UInt`() throws {
    expectNoDifference(try UInt(needleValue: 1), 1)
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  @Test
  func `Initializes Int128`() throws {
    expectNoDifference(try Int128(needleValue: 1), 1)
  }

  @available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
  @Test
  func `Initializes UInt128`() throws {
    expectNoDifference(try UInt128(needleValue: 1), 1)
  }

  @Test
  func `Initializes Data From UTF8 String`() throws {
    expectNoDifference(try Data(needleValue: "blob"), Data("blob".utf8))
  }

  @Test
  func `Initializes Decimal`() throws {
    expectNoDifference(try Decimal(needleValue: 1.5), Decimal(1.5))
    expectNoDifference(try Decimal(needleValue: 1), Decimal(1))
  }

  @Test
  func `Initializes Optional`() throws {
    expectNoDifference(try String?(needleValue: .null), nil)
    expectNoDifference(try String?(needleValue: "blob"), "blob")
  }

  @Test
  func `Initializes Array`() throws {
    expectNoDifference(try [Int](needleValue: [1, 2, 3]), [1, 2, 3])
  }

  @Test
  func `Initializes Dictionary`() throws {
    expectNoDifference(try [String: Int](needleValue: ["one": 1, "two": 2]), ["one": 1, "two": 2])
  }

  @Test
  func `Throws Type Error For Invalid Primitive Value`() {
    #expect(throws: NeedleValueTypeError(expected: .string, received: .integer)) {
      try String(needleValue: 1)
    }
    #expect(throws: NeedleValueTypeError(expected: .boolean, received: .string)) {
      try Bool(needleValue: "blob")
    }
    #expect(throws: NeedleValueTypeError(expected: .number, received: .string)) {
      try Double(needleValue: "blob")
    }
    #expect(throws: NeedleValueTypeError(expected: .integer, received: .number)) {
      try Int(needleValue: 1.5)
    }
    #expect(throws: NeedleValueTypeError(expected: .array, received: .object)) {
      try [String](needleValue: [:])
    }
    #expect(throws: NeedleValueTypeError(expected: .object, received: .array)) {
      try [String: String](needleValue: [])
    }
  }

  @Test
  func `Throws Type Error For Integer Overflow And Underflow`() {
    #expect(throws: NeedleValueTypeError(expected: .integer, received: .integer)) {
      try Int8(needleValue: 999)
    }
    #expect(throws: NeedleValueTypeError(expected: .integer, received: .integer)) {
      try UInt(needleValue: -1)
    }
  }

  @Test
  func `Initializes Macro Generated Type`() throws {
    let value: NeedleValue = [
      "name": "Blob",
      "display_age": 42,
      "nickname": .null,
      "tags": ["swift", "needle"],
      "metadata": ["role": "admin"],
      "address": ["city": "Brooklyn"]
    ]

    let user = try MacroUser(needleValue: value)

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

@NeedleGenerable
private struct MacroUser: Equatable {
  var name: String
  @NeedleGuide(key: "display_age")
  var age: Int
  var nickname: String?
  var tags: [String]
  var metadata: [String: String]
  var address: Address
  @NeedleIgnored
  var ignoredOptional: String?
  @NeedleIgnored
  var ignoredDefault: String = "default"
}

@NeedleGenerable
private struct Address: Equatable {
  var city: String
}
