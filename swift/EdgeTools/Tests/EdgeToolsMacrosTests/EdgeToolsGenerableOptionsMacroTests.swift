import MacroTesting
import Testing

extension `EdgeToolsMacros tests` {
  @Suite
  struct `EdgeToolsGenerableOptionsMacro tests` {
    @Test
    func `Diagnoses Unsupported Type String Storage`() {
      assertMacro {
        """
        @EdgeToolsGenerable(partialStrings: .other)
        struct Message {
          var text: String
        }
        """
      } diagnostics: {
        """
        @EdgeToolsGenerable(partialStrings: .other)
        ┬──────────────────────────────────────────
        ├─ 🛑 partialStrings requires .streamString or .string.
        ╰─ 🛑 partialStrings requires .streamString or .string.
        struct Message {
          var text: String
        }
        """
      }
    }

    @Test
    func `Diagnoses Unsupported Member String Storage`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(partialStrings: .other)
          var text: String
        }
        """
      } diagnostics: {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(partialStrings: .other)
                          ┬─────────────────────
                          ├─ 🛑 partialStrings requires .streamString or .string.
                          ╰─ 🛑 partialStrings requires .streamString or .string.
          var text: String
        }
        """
      }
    }

    @Test
    func `Diagnoses Negative Capacity`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(initialCapacity: -1)
          var text: String
        }
        """
      } diagnostics: {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(initialCapacity: -1)
                          ┬──────────────────
                          ├─ 🛑 initialCapacity must be a nonnegative integer literal.
                          ╰─ 🛑 initialCapacity must be a nonnegative integer literal.
          var text: String
        }
        """
      }
    }

    @Test
    func `Diagnoses Converted Key Collisions`() {
      assertMacro {
        """
        @EdgeToolsGenerable(keyDecodingStrategy: .convertFromSnakeCase)
        struct Message {
          var userID: Int
          var userId: Int
        }
        """
      } diagnostics: {
        """
        @EdgeToolsGenerable(keyDecodingStrategy: .convertFromSnakeCase)
        ┬──────────────────────────────────────────────────────────────
        ╰─ 🛑 The stream key 'user_id' occurs more than once.
        struct Message {
          var userID: Int
          var userId: Int
        }
        """
      }
    }
  }
}
