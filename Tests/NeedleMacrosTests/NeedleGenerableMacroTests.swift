import MacroTesting
import NeedleMacros
import Testing

extension BaseTestSuite {
  @Suite
  struct `NeedleGenerableMacro tests` {
    @Test
    func `Applies Top Level Fragments`() {
      assertMacro {
        """
        @NeedleGenerable(.description("Person payload"))
        struct Person {
          var name: String
        }
        """
      } expansion: {
        """
        struct Person {
          var name: String

          static var needleGenerationSchema: NeedleGenerationSchema {
            NeedleGenerationSchema(
              .type(.object),
                    .description("Person payload"),
                    .properties([
                          "name": String.needleGenerationSchema
                        ]),
                    .required(["name"])
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
    func `Composes Property Fragments`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Person {
          @NeedleGuide(key: "first_name", .minLength(1), .description("Given name"), .examples(["Ada"]))
          var firstName: String
        }
        """
      } expansion: {
        """
        struct Person {
          var firstName: String

          static var needleGenerationSchema: NeedleGenerationSchema {
            NeedleGenerationSchema(
              .type(.object),
                    .properties([
                          "first_name": NeedleGenerationSchema(String.needleGenerationSchema, .minLength(1), .description("Given name"), .examples(["Ada"]))
                        ]),
                    .required(["first_name"])
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
    func `Uses Inferred Optional Schema`() {
      assertMacro {
        """
        @NeedleGenerable
        struct Payload {
          var title: String?
        }
        """
      } expansion: {
        """
        struct Payload {
          var title: String?

          static var needleGenerationSchema: NeedleGenerationSchema {
            NeedleGenerationSchema(
              .type(.object),
                    .properties([
                          "title": String?.needleGenerationSchema
                        ])
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.title = try Optional<String>(needleValue: _needleValue(object, forKey: "title"))
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
            NeedleGenerationSchema(
              .type(.object)
            )
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
    func `Uses Nested NeedleGenerable Objects`() {
      assertMacro {
        """
        struct Address: NeedleGenerable {
          static var needleGenerationSchema: NeedleGenerationSchema {
            NeedleGenerationSchema(.type(.object))
          }

          init(needleValue: NeedleValue) throws {}
        }
        @NeedleGenerable
        struct Person {
          var address: Address
        }
        """
      } expansion: {
        """
        struct Address: NeedleGenerable {
          static var needleGenerationSchema: NeedleGenerationSchema {
            NeedleGenerationSchema(.type(.object))
          }

          init(needleValue: NeedleValue) throws {}
        }
        struct Person {
          var address: Address

          static var needleGenerationSchema: NeedleGenerationSchema {
            NeedleGenerationSchema(
              .type(.object),
                    .properties([
                          "address": Address.needleGenerationSchema
                        ]),
                    .required(["address"])
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.address = try Address(needleValue: _needleValue(object, forKey: "address"))
          }
        }

        extension Person: NeedleGenerable {
        }
        """
      }
    }

    @Test
    func `Uses Arrays Of NeedleGenerable Objects`() {
      assertMacro {
        """
        struct Address: NeedleGenerable {
          static var needleGenerationSchema: NeedleGenerationSchema {
            NeedleGenerationSchema(.type(.object))
          }

          init(needleValue: NeedleValue) throws {}
        }
        @NeedleGenerable
        struct Person {
          var addresses: [Address]
        }
        """
      } expansion: {
        """
        struct Address: NeedleGenerable {
          static var needleGenerationSchema: NeedleGenerationSchema {
            NeedleGenerationSchema(.type(.object))
          }

          init(needleValue: NeedleValue) throws {}
        }
        struct Person {
          var addresses: [Address]

          static var needleGenerationSchema: NeedleGenerationSchema {
            NeedleGenerationSchema(
              .type(.object),
                    .properties([
                          "addresses": [Address].needleGenerationSchema
                        ]),
                    .required(["addresses"])
            )
          }

          init(needleValue: NeedleValue) throws {
            let object = try _needleRequireObjectValue(needleValue)
            self.addresses = try [Address](needleValue: _needleValue(object, forKey: "addresses"))
          }
        }

        extension Person: NeedleGenerable {
        }
        """
      }
    }
  }
}
