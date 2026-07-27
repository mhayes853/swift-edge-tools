import EdgeToolsMacros
import MacroTesting
import Testing

extension BaseTestSuite {
  @Suite
  struct `EdgeToolsGenerableMacro tests` {
    @Test
    func `Applies Top Level Fragments`() {
      assertMacro {
        """
        @EdgeToolsGenerable(.description("Person payload"))
        struct Person {
          var name: String
        }
        """
      } expansion: {
        """
        struct Person {
          var name: String

          static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
            EdgeToolsGenerationSchema(
              .type(.object),
                    .description("Person payload"),
                    .properties([
                          "name": String.edgeToolsGenerationSchema
                        ]),
                    .required(["name"])
            )
          }

          init(edgeToolsValue: EdgeToolsValue) throws {
            let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
            self.name = try String(edgeToolsValue: _edgeToolsValue(object, forKey: "name"))
          }

          var edgeToolsValue: EdgeToolsValue {
            _edgeToolsBuildObjectValue(
              (key: "name", value: self.name.edgeToolsValue)
            )
          }
        }

        extension Person: EdgeToolsGenerable {
        }
        """
      }
    }

    @Test
    func `Composes Property Fragments`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Person {
          @EdgeToolsGuide(key: "first_name", .minLength(1), .description("Given name"), .examples(["Ada"]))
          var firstName: String
        }
        """
      } expansion: {
        """
        struct Person {
          var firstName: String

          static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
            EdgeToolsGenerationSchema(
              .type(.object),
                    .properties([
                          "first_name": EdgeToolsGenerationSchema(String.edgeToolsGenerationSchema, .minLength(1), .description("Given name"), .examples(["Ada"]))
                        ]),
                    .required(["first_name"])
            )
          }

          init(edgeToolsValue: EdgeToolsValue) throws {
            let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
            self.firstName = try String(edgeToolsValue: _edgeToolsValue(object, forKey: "first_name"))
          }

          var edgeToolsValue: EdgeToolsValue {
            _edgeToolsBuildObjectValue(
              (key: "first_name", value: self.firstName.edgeToolsValue)
            )
          }
        }

        extension Person: EdgeToolsGenerable {
        }
        """
      }
    }

    @Test
    func `Includes Stored Properties With Observers`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Person {
          var name: String {
            didSet {}
          }
        }
        """
      } expansion: {
        """
        struct Person {
          var name: String {
            didSet {}
          }

          static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
            EdgeToolsGenerationSchema(
              .type(.object),
                    .properties([
                          "name": String.edgeToolsGenerationSchema
                        ]),
                    .required(["name"])
            )
          }

          init(edgeToolsValue: EdgeToolsValue) throws {
            let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
            self.name = try String(edgeToolsValue: _edgeToolsValue(object, forKey: "name"))
          }

          var edgeToolsValue: EdgeToolsValue {
            _edgeToolsBuildObjectValue(
              (key: "name", value: self.name.edgeToolsValue)
            )
          }
        }

        extension Person: EdgeToolsGenerable {
        }
        """
      }
    }

    @Test
    func `Uses Inferred Optional Schema`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Payload {
          var title: String?
        }
        """
      } expansion: {
        """
        struct Payload {
          var title: String?

          static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
            EdgeToolsGenerationSchema(
              .type(.object),
                    .properties([
                          "title": String?.edgeToolsGenerationSchema
                        ])
            )
          }

          init(edgeToolsValue: EdgeToolsValue) throws {
            let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
            self.title = try Optional<String>(edgeToolsValue: _edgeToolsValue(object, forKey: "title"))
          }

          var edgeToolsValue: EdgeToolsValue {
            _edgeToolsBuildObjectValue(
              (key: "title", value: self.title?.edgeToolsValue)
            )
          }
        }

        extension Payload: EdgeToolsGenerable {
        }
        """
      }
    }

    @Test
    func `Decodes Ignored Optional As Nil`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Payload {
          @EdgeToolsIgnored
          var internalID: String?
        }
        """
      } expansion: {
        """
        struct Payload {
          var internalID: String?

          static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
            EdgeToolsGenerationSchema(
              .type(.object)
            )
          }

          init(edgeToolsValue: EdgeToolsValue) throws {
            let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
            self.internalID = nil
          }

          var edgeToolsValue: EdgeToolsValue {
            _edgeToolsBuildObjectValue()
          }
        }

        extension Payload: EdgeToolsGenerable {
        }
        """
      }
    }

    @Test
    func `Uses Nested EdgeToolsGenerable Objects`() {
      assertMacro {
        """
        struct Address: EdgeToolsGenerable {
          static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
            EdgeToolsGenerationSchema(.type(.object))
          }

          init(edgeToolsValue: EdgeToolsValue) throws {}
        }
        @EdgeToolsGenerable
        struct Person {
          var address: Address
        }
        """
      } expansion: {
        """
        struct Address: EdgeToolsGenerable {
          static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
            EdgeToolsGenerationSchema(.type(.object))
          }

          init(edgeToolsValue: EdgeToolsValue) throws {}
        }
        struct Person {
          var address: Address

          static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
            EdgeToolsGenerationSchema(
              .type(.object),
                    .properties([
                          "address": Address.edgeToolsGenerationSchema
                        ]),
                    .required(["address"])
            )
          }

          init(edgeToolsValue: EdgeToolsValue) throws {
            let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
            self.address = try Address(edgeToolsValue: _edgeToolsValue(object, forKey: "address"))
          }

          var edgeToolsValue: EdgeToolsValue {
            _edgeToolsBuildObjectValue(
              (key: "address", value: self.address.edgeToolsValue)
            )
          }
        }

        extension Person: EdgeToolsGenerable {
        }
        """
      }
    }

    @Test
    func `Uses Arrays Of EdgeToolsGenerable Objects`() {
      assertMacro {
        """
        struct Address: EdgeToolsGenerable {
          static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
            EdgeToolsGenerationSchema(.type(.object))
          }

          init(edgeToolsValue: EdgeToolsValue) throws {}
        }
        @EdgeToolsGenerable
        struct Person {
          var addresses: [Address]
        }
        """
      } expansion: {
        """
        struct Address: EdgeToolsGenerable {
          static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
            EdgeToolsGenerationSchema(.type(.object))
          }

          init(edgeToolsValue: EdgeToolsValue) throws {}
        }
        struct Person {
          var addresses: [Address]

          static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
            EdgeToolsGenerationSchema(
              .type(.object),
                    .properties([
                          "addresses": [Address].edgeToolsGenerationSchema
                        ]),
                    .required(["addresses"])
            )
          }

          init(edgeToolsValue: EdgeToolsValue) throws {
            let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
            self.addresses = try [Address](edgeToolsValue: _edgeToolsValue(object, forKey: "addresses"))
          }

          var edgeToolsValue: EdgeToolsValue {
            _edgeToolsBuildObjectValue(
              (key: "addresses", value: self.addresses.edgeToolsValue)
            )
          }
        }

        extension Person: EdgeToolsGenerable {
        }
        """
      }
    }
  }
}
