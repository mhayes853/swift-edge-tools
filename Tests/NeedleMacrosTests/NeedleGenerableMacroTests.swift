import MacroTesting
import NeedleMacros
import Testing

extension BaseTestSuite {
  @Suite
  struct `NeedleGenerableMacro tests` {
    @Test
    func `Applies Top Level Description`() {
      assertMacro {
        """
        @NeedleGenerable(description: "Person payload")
        struct Person {
          var name: String
        }
        """
      } expansion: {
        """
        struct Person {
          var name: String

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(
              description: "Person payload",
              valueSchema: .object(
                properties: [
                  "name": String.needleGenerationSchema
                ],
                required: ["name"]
              )
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.name = try String(needleValue: _needleValue(object, forKey: "name"))
          }
        }

        extension Person: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Expands Inner NeedleGenerable Type Using Qualified Extension Name`() {
      assertMacro {
        """
        struct Outer {
          @NeedleGenerable
          struct Inner {
            var name: String
          }
        }
        """
      } expansion: {
        """
        struct Outer {
          struct Inner {
            var name: String

            static var needleGenerationSchema: NeedleGenerationSchema {
              .object(
                valueSchema: .object(
                  properties: [
                    "name": String.needleGenerationSchema
                  ],
                  required: ["name"]
                )
              )
            }

            init(needleValue: NeedleValue) throws {
              let object = try _needleRequireObjectValue(needleValue)
              self.name = try String(needleValue: _needleValue(object, forKey: "name"))
            }
          }
        }

        extension Outer.Inner: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Applies Inferred Property Description`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Person {
          @NeedleGuide(description: "Given name")
          var name: String
        }
        """
      } expansion: {
        """
        struct Person {
          var name: String

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(
              valueSchema: .object(
                properties: [
                  "name": _needleMergeGenerationSchema(String.needleGenerationSchema, description: "Given name")
                ],
                required: ["name"]
              )
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.name = try String(needleValue: _needleValue(object, forKey: "name"))
          }
        }

        extension Person: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Uses Merge Helper For Property Description`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Person {
          @NeedleGuide(description: "Display name")
          var name: String
        }
        """
      } expansion: {
        """
        struct Person {
          var name: String

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(
              valueSchema: .object(
                properties: [
                  "name": _needleMergeGenerationSchema(String.needleGenerationSchema, description: "Display name")
                ],
                required: ["name"]
              )
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.name = try String(needleValue: _needleValue(object, forKey: "name"))
          }
        }

        extension Person: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Applies Key Description And String Schema`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Person {
          @NeedleGuide(.string(minLength: 1, maxLength: 10), key: "first_name", description: "Given name")
          var firstName: String
        }
        """
      } expansion: {
        """
        struct Person {
          var firstName: String

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(
              valueSchema: .object(
                properties: [
                  "first_name": _needleMergeGenerationSchema(.string(minLength: 1, maxLength: 10), description: "Given name")
                ],
                required: ["first_name"]
              )
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.firstName = try String(needleValue: _needleValue(object, forKey: "first_name"))
          }
        }

        extension Person: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Auto Applies StreamParseableMember For Keyed NeedleGuide`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(key: "display_name")
          var name: String
        }
        """
      } expansion: {
        """
        struct Payload {
          var name: String

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(
              valueSchema: .object(
                properties: [
                  "display_name": String.needleGenerationSchema
                ],
                required: ["display_name"]
              )
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.name = try String(needleValue: _needleValue(object, forKey: "display_name"))
          }
        }

        extension Payload: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Decodes Ignored Optional As Nil`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleIgnored
          var internalID: String?
        }
        """
      } expansion: {
        """
        struct Payload {
          var internalID: String?

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(valueSchema: .object())
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.internalID = nil
          }
        }

        extension Payload: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Does Not Auto Apply StreamParseableMember When Already Applied`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(key: "display_name")
          @StreamParseableMember(key: "name")
          var name: String
        }
        """
      } expansion: {
        """
        struct Payload {
          @StreamParseableMember(key: "name")
          var name: String

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(
              valueSchema: .object(
                properties: [
                  "display_name": String.needleGenerationSchema
                ],
                required: ["display_name"]
              )
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.name = try String(needleValue: _needleValue(object, forKey: "display_name"))
          }
        }

        extension Payload: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Does Not Auto Apply StreamParseableIgnored When Already Applied`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleIgnored
          @StreamParseableIgnored
          var internalID: String?
        }
        """
      } expansion: {
        """
        struct Payload {
          @StreamParseableIgnored
          var internalID: String?

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(valueSchema: .object())
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.internalID = nil
          }
        }

        extension Payload: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Accepts Regex Literal Pattern In String Schema`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Person {
          @NeedleGuide(.string(pattern: /[a-z]+/))
          var name: String
        }
        """
      } expansion: {
        """
        struct Person {
          var name: String

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(
              valueSchema: .object(
                properties: [
                  "name": .string(pattern: "[a-z]+")
                ],
                required: ["name"]
              )
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.name = try String(needleValue: _needleValue(object, forKey: "name"))
          }
        }

        extension Person: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Accepts Hash Regex Literal Pattern In String Schema`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Person {
          @NeedleGuide(.string(pattern: #/[a-z]+/#))
          var name: String
        }
        """
      } expansion: {
        """
        struct Person {
          var name: String

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(
              valueSchema: .object(
                properties: [
                  "name": .string(pattern: #"[a-z]+"#)
                ],
                required: ["name"]
              )
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.name = try String(needleValue: _needleValue(object, forKey: "name"))
          }
        }

        extension Person: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Applies Boolean And Number Schema For Optional Properties`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.number(minimum: 0, exclusiveMaximum: 1))
          var confidence: Double?
          @NeedleGuide(.boolean)
          var isVisible: Bool?
        }
        """
      } expansion: {
        """
        struct Payload {
          var confidence: Double?
          var isVisible: Bool?

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(
              valueSchema: .object(
                properties: [
                  "confidence": .union(number: .number(minimum: 0, exclusiveMaximum: 1), null: true),
                    "isVisible": .union(bool: true, null: true)
                ]
              )
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.confidence = try Optional<Double>(needleValue: _needleValue(object, forKey: "confidence"))
            self.isVisible = try Optional<Bool>(needleValue: _needleValue(object, forKey: "isVisible"))
          }
        }

        extension Payload: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Applies Array And Object Schema`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.array(minItems: 1, uniqueItems: true))
          var tags: [String]
          @NeedleGuide(.object(minProperties: 1))
          var metadata: [String: Int]
        }
        """
      } expansion: {
        """
        struct Payload {
          var tags: [String]
          var metadata: [String: Int]

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(
              valueSchema: .object(
                properties: [
                  "tags": .array(items: .schemaForAll(String.needleGenerationSchema), minItems: 1, uniqueItems: true),
                    "metadata": .object(minProperties: 1, additionalProperties: Int.needleGenerationSchema)
                ],
                required: ["tags", "metadata"]
              )
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.tags = try [String](needleValue: _needleValue(object, forKey: "tags"))
            self.metadata = try [String: Int](needleValue: _needleValue(object, forKey: "metadata"))
          }
        }

        extension Payload: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Rejects NeedleIgnored Combined With NeedleGuide`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleIgnored
          @NeedleGuide(.string(minLength: 1))
          var name: String
        }
        """
      } diagnostics: {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleIgnored
          @NeedleGuide(.string(minLength: 1))
          ┬──────────────────────────────────
          ╰─ 🛑 @NeedleIgnored cannot be combined with @NeedleGuide on the same property.
          var name: String
        }
        """
      }
    }

    @Test
    func `Rejects Multiple NeedleGuide Attributes On One Property`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(key: "display_name")
          @NeedleGuide(key: "name")
          var name: String
        }
        """
      } diagnostics: {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(key: "display_name")
          @NeedleGuide(key: "name")
          ┬────────────────────────
          ╰─ 🛑 Only one @NeedleGuide attribute can be applied to a stored property.
          var name: String
        }
        """
      }
    }

    @Test
    func `Rejects String Schema On Non String Property`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.string(minLength: 1))
          var age: Int
        }
        """
      } diagnostics: {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.string(minLength: 1))
          ┬──────────────────────────────────
          ╰─ 🛑 @NeedleGuide(.string) can only be applied to properties of type String. Found 'age: Int'.
          var age: Int
        }
        """
      }
    }

    @Test
    func `Rejects Number Schema On Non Number Property`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.number(minimum: 0))
          var slug: String
        }
        """
      } diagnostics: {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.number(minimum: 0))
          ┬────────────────────────────────
          ╰─ 🛑 @NeedleGuide(.number) can only be applied to properties of type number (Double, Float, CGFloat, Decimal). Found 'slug: String'.
          var slug: String
        }
        """
      }
    }

    @Test
    func `Rejects Integer Schema On Non Integer Property`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.integer(minimum: 1))
          var price: Double
        }
        """
      } diagnostics: {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.integer(minimum: 1))
          ┬─────────────────────────────────
          ╰─ 🛑 @NeedleGuide(.integer) can only be applied to properties of type integer (Int, Int8, Int16, Int32, Int64, UInt, UInt8, UInt16, UInt32, UInt64, Int128, UInt128). Found 'price: Double'.
          var price: Double
        }
        """
      }
    }

    @Test
    func `Rejects Boolean Schema On Non Boolean Property`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.boolean)
          var title: String
        }
        """
      } diagnostics: {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.boolean)
          ┬─────────────────────
          ╰─ 🛑 @NeedleGuide(.boolean) can only be applied to properties of type Bool. Found 'title: String'.
          var title: String
        }
        """
      }
    }

    @Test
    func `Rejects Array Schema On Non Array Property`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.array(minItems: 1))
          var name: String
        }
        """
      } diagnostics: {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.array(minItems: 1))
          ┬────────────────────────────────
          ╰─ 🛑 @NeedleGuide(.array) can only be applied to array properties. Found 'name: String'.
          var name: String
        }
        """
      }
    }

    @Test
    func `Rejects Object Schema On Non Dictionary Property`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.object(minProperties: 1))
          var name: String
        }
        """
      } diagnostics: {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.object(minProperties: 1))
          ┬──────────────────────────────────────
          ╰─ 🛑 @NeedleGuide(.object) can only be applied to dictionary properties. Found 'name: String'.
          var name: String
        }
        """
      }
    }

    @Test
    func `Rejects Object Schema On Non String Key Dictionary`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.object(minProperties: 1))
          var byID: [Int: String]
        }
        """
      } diagnostics: {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.object(minProperties: 1))
          ┬──────────────────────────────────────
          ╰─ 🛑 @NeedleGuide(.object) can only be applied to dictionary properties with String keys. Found 'byID: [Int: String]'.
          var byID: [Int: String]
        }
        """
      }
    }

    @Test
    func `Allows Primitive Schemas On Optional Counterparts`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.string(minLength: 1))
          var title: String?
          @NeedleGuide(.integer(minimum: 1))
          var count: Int?
        }
        """
      } expansion: {
        """
        struct Payload {
          var title: String?
          var count: Int?

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(
              valueSchema: .object(
                properties: [
                  "title": .union(string: .string(minLength: 1), null: true),
                    "count": .union(integer: .integer(minimum: 1), null: true)
                ]
              )
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.title = try Optional<String>(needleValue: _needleValue(object, forKey: "title"))
            self.count = try Optional<Int>(needleValue: _needleValue(object, forKey: "count"))
          }
        }

        extension Payload: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Rejects Array String Schema For Int Elements`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.array(items: .schemaForAll(.string(minLength: 1))))
          var values: [Int]
        }
        """
      } diagnostics: {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.array(items: .schemaForAll(.string(minLength: 1))))
          ┬────────────────────────────────────────────────────────────────
          ╰─ 🛑 @NeedleGuide(.string) can only be applied to properties of type String. Found 'values: Int'.
          var values: [Int]
        }
        """
      }
    }

    @Test
    func `Rejects Object Integer Schema For String Values`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.object(additionalProperties: .integer(minimum: 0)))
          var map: [String: String]
        }
        """
      } diagnostics: {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.object(additionalProperties: .integer(minimum: 0)))
          ┬────────────────────────────────────────────────────────────────
          ╰─ 🛑 @NeedleGuide(.integer) can only be applied to properties of type integer (Int, Int8, Int16, Int32, Int64, UInt, UInt8, UInt16, UInt32, UInt64, Int128, UInt128). Found 'map: String'.
          var map: [String: String]
        }
        """
      }
    }

    @Test
    func `Allows Nested Array Schema At Second Level`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.array(items: .schemaForAll(.array(items: .schemaForAll(.string(minLength: 1))))))
          var values: [[String]]
        }
        """
      } expansion: {
        """
        struct Payload {
          var values: [[String]]

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(
              valueSchema: .object(
                properties: [
                  "values": .array(items: .schemaForAll(.array(items: .schemaForAll(.string(minLength: 1)))))
                ],
                required: ["values"]
              )
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.values = try [[String]](needleValue: _needleValue(object, forKey: "values"))
          }
        }

        extension Payload: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Rejects Nested Array Schema At Second Level`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.array(items: .schemaForAll(.array(items: .schemaForAll(.string(minLength: 1))))))
          var values: [[Int]]
        }
        """
      } diagnostics: {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.array(items: .schemaForAll(.array(items: .schemaForAll(.string(minLength: 1))))))
          ┬──────────────────────────────────────────────────────────────────────────────────────────────
          ╰─ 🛑 @NeedleGuide(.string) can only be applied to properties of type String. Found 'values: Int'.
          var values: [[Int]]
        }
        """
      }
    }

    @Test
    func `Allows Deep DictionaryArrayDictionary`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.object(additionalProperties: .object(additionalProperties: .object(additionalProperties: .integer(minimum: 0)))))
          var payload: [String: [String: [String: Int]]]
        }
        """
      } expansion: {
        """
        struct Payload {
          var payload: [String: [String: [String: Int]]]

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(
              valueSchema: .object(
                properties: [
                  "payload": .object(additionalProperties: .object(additionalProperties: .object(additionalProperties: .integer(minimum: 0))))
                ],
                required: ["payload"]
              )
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.payload = try [String: [String: [String: Int]]](needleValue: _needleValue(object, forKey: "payload"))
          }
        }

        extension Payload: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Rejects Deep DictionaryArrayDictionary Mismatch`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.object(additionalProperties: .object(additionalProperties: .object(additionalProperties: .string(minLength: 1)))))
          var payload: [String: [String: [String: Int]]]
        }
        """
      } diagnostics: {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.object(additionalProperties: .object(additionalProperties: .object(additionalProperties: .string(minLength: 1)))))
          ┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
          ╰─ 🛑 @NeedleGuide(.string) can only be applied to properties of type String. Found 'payload: Int'.
          var payload: [String: [String: [String: Int]]]
        }
        """
      }
    }

    @Test
    func `Rejects Nested Dictionary With Non String Inner Keys`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.object(additionalProperties: .object(additionalProperties: .integer(minimum: 0))))
          var payload: [String: [Int: Int]]
        }
        """
      } diagnostics: {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.object(additionalProperties: .object(additionalProperties: .integer(minimum: 0))))
          ┬───────────────────────────────────────────────────────────────────────────────────────────────
          ╰─ 🛑 @NeedleGuide(.object) can only be applied to dictionary properties with String keys. Found 'payload: [Int:Int]'.
          var payload: [String: [Int: Int]]
        }
        """
      }
    }

    @Test
    func `Rejects Deep TripleArray Primitive Mismatch`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.array(items: .schemaForAll(.array(items: .schemaForAll(.array(items: .schemaForAll(.string(minLength: 1))))))))
          var values: [[[Int]]]
        }
        """
      } diagnostics: {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleGuide(.array(items: .schemaForAll(.array(items: .schemaForAll(.array(items: .schemaForAll(.string(minLength: 1))))))))
          ┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
          ╰─ 🛑 @NeedleGuide(.string) can only be applied to properties of type String. Found 'values: Int'.
          var values: [[[Int]]]
        }
        """
      }
    }

    @Test
    func `Uses Default Value For Ignored Defaulted Property`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleIgnored
          var internalID: String = "default"
        }
        """
      } expansion: {
        """
        struct Payload {
          var internalID: String = "default"

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(valueSchema: .object())
          }

          init(needleValue: NeedleValue) throws {
            _ = try _needleRequireObjectValue(needleValue)
          }
        }

        extension Payload: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Rejects Ignored Non Optional Property Without Default`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleIgnored
          var internalID: String
        }
        """
      } diagnostics: {
        """
        @NeedleGenerable
        struct Payload {
          @NeedleIgnored
          ┬─────────────
          ╰─ 🛑 @NeedleIgnored can only be applied to optional properties or properties with default values.
          var internalID: String
        }
        """
      }
    }

    @Test
    func `Does Not Generate Needle Value Initializer When Already Declared`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          var name: String

          init(needleValue: NeedleValue) throws {
            self.name = "custom"
          }
        }
        """
      } expansion: {
        """
        struct Payload {
          var name: String

          init(needleValue: NeedleValue) throws {
            self.name = "custom"
          }

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(
              valueSchema: .object(
                properties: [
                  "name": String.needleGenerationSchema
                ],
                required: ["name"]
              )
            )
          }
        }

        extension Payload: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Initializes Nested NeedleGenerable Property`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          var user: User
        }

        @NeedleGenerable
        struct User {
          var name: String
        }
        """
      } expansion: {
        """
        struct Payload {
          var user: User

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(
              valueSchema: .object(
                properties: [
                  "user": User.needleGenerationSchema
                ],
                required: ["user"]
              )
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.user = try User(needleValue: _needleValue(object, forKey: "user"))
          }
        }
        struct User {
          var name: String

          static var needleGenerationSchema: NeedleGenerationSchema {
            .object(
              valueSchema: .object(
                properties: [
                  "name": String.needleGenerationSchema
                ],
                required: ["name"]
              )
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.name = try String(needleValue: _needleValue(object, forKey: "name"))
          }
        }

        extension Payload: NeedleGenerable {
        }

        extension User: NeedleGenerable {
        }
        """
      }
    }
  }
}
