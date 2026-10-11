import MacroTesting
import Testing

extension `EdgeToolsMacros tests` {
  @Suite
  struct `EdgeToolsKeyAliasMacro tests` {
    @Test
    func `Generates Aliases With A Custom Partial`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Message {
          typealias Partial = String
          @EdgeToolsGuide(keyNames: ["text", "legacy_text"])
          var text: String
        }
        """
      } expansion: {
        """
        struct Message {
          typealias Partial = String
          var text: String

          static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
            EdgeToolsGenerationSchema(
              .type(.object),
                    .properties([
                          "text": String.edgeToolsGenerationSchema
                        ]),
                    .required(["text"])
            )
          }

          init(edgeToolsValue: EdgeToolsValue) throws {
            let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
            self.text = try String(edgeToolsValue: (_edgeToolsValue(object, forKeys: ["text", "legacy_text"]) ?? .null))
          }

          var edgeToolsValue: EdgeToolsValue {
            _edgeToolsBuildObjectValue(
              (key: "text", value: self.text.edgeToolsValue)
            )
          }
        }

        extension Message: EdgeToolsGenerable {
        }
        """
      }
    }

    @Test
    func `Diagnoses Conflicting Key Options`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(key: "text", keyNames: ["old_text"])
          var text: String
        }
        """
      } diagnostics: {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(key: "text", keyNames: ["old_text"])
          ┬───────────────────────────────────────────────────
          ╰─ 🛑 @EdgeToolsGuide takes either key: or keyNames:, not both.
          var text: String
        }
        """
      }
    }

    @Test
    func `Diagnoses Empty Key Lists`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(keyNames: [])
          var text: String
        }
        """
      } diagnostics: {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(keyNames: [])
                                    ┬─
                                    ╰─ 🛑 @EdgeToolsGuide(keyNames:) requires a string array literal.
          var text: String
        }
        """
      }
    }

    @Test
    func `Diagnoses Empty Aliases`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(keyNames: ["text", ""])
          var text: String
        }
        """
      } diagnostics: {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(keyNames: ["text", ""])
                                    ┬───────────
                                    ╰─ 🛑 @EdgeToolsGuide(keyNames:) must not contain an empty name.
          var text: String
        }
        """
      }
    }

    @Test
    func `Diagnoses Interpolated Aliases`() {
      assertMacro {
        #"""
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(keyNames: ["text", "old_\(1)"])
          var text: String
        }
        """#
      } diagnostics: {
        #"""
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(keyNames: ["text", "old_\(1)"])
                                    ┬───────────────────
                                    ╰─ 🛑 @EdgeToolsGuide(keyNames:) requires a string array literal.
          var text: String
        }
        """#
      }
    }

    @Test
    func `Diagnoses Aliases Claimed By Another Field`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(keyNames: ["text", "legacy"])
          var text: String
          @EdgeToolsGuide(key: "legacy")
          var other: String
        }
        """
      } diagnostics: {
        """
        @EdgeToolsGenerable
        ┬──────────────────
        ╰─ 🛑 The stream key 'legacy' occurs more than once.
        struct Message {
          @EdgeToolsGuide(keyNames: ["text", "legacy"])
          var text: String
          @EdgeToolsGuide(key: "legacy")
          var other: String
        }
        """
      }
    }

    @Test
    func `Diagnoses Repeated Aliases`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(keyNames: ["text", "text"])
          var text: String
        }
        """
      } diagnostics: {
        """
        @EdgeToolsGenerable
        ┬──────────────────
        ╰─ 🛑 The stream key 'text' occurs more than once.
        struct Message {
          @EdgeToolsGuide(keyNames: ["text", "text"])
          var text: String
        }
        """
      }
    }
  }
}
