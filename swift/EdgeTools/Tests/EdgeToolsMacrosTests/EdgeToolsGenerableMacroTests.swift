import EdgeToolsMacros
import MacroTesting
import Testing

extension `EdgeToolsMacros tests` {
  @Suite
  struct `EdgeToolsGenerableMacro tests` {
    @Test
    func `Generates Associated Value Enum Conformance`() {
      assertMacro {
        """
        @EdgeToolsGenerable(.description("An action"))
        enum Action {
          case move(Double, Double)
          case search(query: String, limit: Int?)
        }
        """
      } expansion: {
        """
        enum Action {
          case move(Double, Double)
          case search(query: String, limit: Int?)

          static var extractionToolDefinition: EdgeToolDefinition {
            EdgeToolDefinition(
              name: "action",
              description: edgeToolsGenerationSchema.objectValue? [.description]?.string
                ?? "Extract structured data from the input.",
              arguments: edgeToolsGenerationSchema
            )
          }

          static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
            EdgeToolsGenerationSchema(
              .anyOf([
                      EdgeToolsGenerationSchema(
            .type(.object),
            .properties([
              "move": EdgeToolsGenerationSchema(
                .type(.object),
                .properties([
                  "_0": Double.edgeToolsGenerationSchema,
                                "_1": Double.edgeToolsGenerationSchema
                ]),
                .required(["_0", "_1"]),
                .additionalProperties(false)
              )
            ]),
            .required(["move"]),
            .additionalProperties(false)
                            ),
                      EdgeToolsGenerationSchema(
            .type(.object),
            .properties([
              "search": EdgeToolsGenerationSchema(
                .type(.object),
                .properties([
                  "query": String.edgeToolsGenerationSchema,
                                "limit": Int?.edgeToolsGenerationSchema
                ]),
                .required(["query", "limit"]),
                .additionalProperties(false)
              )
            ]),
            .required(["search"]),
            .additionalProperties(false)
                            )
                    ]),
                    .description("An action")
            )
          }

          init(edgeToolsValue: EdgeToolsValue) throws {
            let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
            if let value = object["move"] {
            let payload = try _edgeToolsRequireObjectValue(value, keys: ["_0", "_1"])
            self = .move(
              try Double(edgeToolsValue: _edgeToolsValue(payload, forKey: "_0")),
                    try Double(edgeToolsValue: _edgeToolsValue(payload, forKey: "_1"))
            )
            return
            }
            if let value = object["search"] {
              let payload = try _edgeToolsRequireObjectValue(value, keys: ["query", "limit"])
              self = .search(
                query: try String(edgeToolsValue: _edgeToolsValue(payload, forKey: "query")),
                      limit: try Optional<Int>(edgeToolsValue: _edgeToolsValue(payload, forKey: "limit"))
              )
              return
            }
            throw EdgeToolsUnknownEnumCaseError(
              typeName: "Action",
              caseName: object.keys.first ?? ""
            )
          }

          var edgeToolsValue: EdgeToolsValue {
            switch self {
            case .move(let value0, let value1):
            _edgeToolsBuildObjectValue(
              (key: "move", value: _edgeToolsBuildObjectValue(
                (key: "_0", value: value0.edgeToolsValue),
                      (key: "_1", value: value1.edgeToolsValue)
              )))
            case .search(let value0, let value1):
              _edgeToolsBuildObjectValue(
                (key: "search", value: _edgeToolsBuildObjectValue(
                  (key: "query", value: value0.edgeToolsValue),
                        (key: "limit", value: value1.edgeToolsValue)
                )))
            }
          }
        }

        extension Action: EdgeToolsGenerable {
        }
        """
      }
    }

    @Test
    func `Requires Enum Associated Values`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        enum Status {
          case ready
        }
        """
      } diagnostics: {
        """
        @EdgeToolsGenerable
        ┬──────────────────
        ╰─ 🛑 @EdgeToolsGenerable enum case 'ready' must have at least one associated value.
        enum Status {
          case ready
        }
        """
      }
    }

    @Test
    func `Rejects Ambiguous Enum Keys`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        enum Action {
          case load(id: Int)
          case load(name: String)
        }

        @EdgeToolsGenerable
        enum Collision {
          case value(_1: Int, String)
        }
        """
      } diagnostics: {
        """
        @EdgeToolsGenerable
        ┬──────────────────
        ╰─ 🛑 @EdgeToolsGenerable does not support overloaded enum case names ('load').
        enum Action {
          case load(id: Int)
          case load(name: String)
        }

        @EdgeToolsGenerable
        ┬──────────────────
        ╰─ 🛑 Enum case 'value' has multiple associated values represented by the key '_1'.
        enum Collision {
          case value(_1: Int, String)
        }
        """
      }
    }

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

          static var extractionToolDefinition: EdgeToolDefinition {
            EdgeToolDefinition(
              name: "person",
              description: edgeToolsGenerationSchema.objectValue? [.description]?.string
                ?? "Extract structured data from the input.",
              arguments: edgeToolsGenerationSchema
            )
          }

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

          static var extractionToolDefinition: EdgeToolDefinition {
            EdgeToolDefinition(
              name: "person",
              description: edgeToolsGenerationSchema.objectValue? [.description]?.string
                ?? "Extract structured data from the input.",
              arguments: edgeToolsGenerationSchema
            )
          }

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

          static var extractionToolDefinition: EdgeToolDefinition {
            EdgeToolDefinition(
              name: "person",
              description: edgeToolsGenerationSchema.objectValue? [.description]?.string
                ?? "Extract structured data from the input.",
              arguments: edgeToolsGenerationSchema
            )
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

          static var extractionToolDefinition: EdgeToolDefinition {
            EdgeToolDefinition(
              name: "payload",
              description: edgeToolsGenerationSchema.objectValue? [.description]?.string
                ?? "Extract structured data from the input.",
              arguments: edgeToolsGenerationSchema
            )
          }

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

          static var extractionToolDefinition: EdgeToolDefinition {
            EdgeToolDefinition(
              name: "payload",
              description: edgeToolsGenerationSchema.objectValue? [.description]?.string
                ?? "Extract structured data from the input.",
              arguments: edgeToolsGenerationSchema
            )
          }

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

          static var extractionToolDefinition: EdgeToolDefinition {
            EdgeToolDefinition(
              name: "person",
              description: edgeToolsGenerationSchema.objectValue? [.description]?.string
                ?? "Extract structured data from the input.",
              arguments: edgeToolsGenerationSchema
            )
          }

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

          static var extractionToolDefinition: EdgeToolDefinition {
            EdgeToolDefinition(
              name: "person",
              description: edgeToolsGenerationSchema.objectValue? [.description]?.string
                ?? "Extract structured data from the input.",
              arguments: edgeToolsGenerationSchema
            )
          }

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
