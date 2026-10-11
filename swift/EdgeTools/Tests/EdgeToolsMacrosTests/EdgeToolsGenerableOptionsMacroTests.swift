import MacroTesting
import Testing

extension `EdgeToolsMacros tests` {
  @Suite
  struct `EdgeToolsGenerableOptionsMacro tests` {
    @Test
    func `Diagnoses Empty Guide Keys`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(key: "")
          var text: String
        }
        """
      } diagnostics: {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(key: "")
                               ┬─
                               ╰─ 🛑 @EdgeToolsGuide(key:) must not be empty.
          var text: String
        }
        """
      }
    }

    @Test
    func `Diagnoses Interpolated Guide Keys`() {
      assertMacro {
        #"""
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(key: "prefix\(1)")
          var text: String
        }
        """#
      } diagnostics: {
        #"""
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(key: "prefix\(1)")
                               ┬───────────
                               ╰─ 🛑 @EdgeToolsGuide(key:) requires a string literal.
          var text: String
        }
        """#
      }
    }

    @Test
    func `Diagnoses Capacity Overflow`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(initialCapacity: 0xFFFF_FFFF_FFFF_FFFF)
          var text: String
        }
        """
      } diagnostics: {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(initialCapacity: 0xFFFF_FFFF_FFFF_FFFF)
                                           ┬────────────────────
                                           ╰─ 🛑 @EdgeToolsGuide(initialCapacity:) requires a nonnegative integer literal.
          var text: String
        }
        """
      }
    }

    @Test
    func `Diagnoses A Conversion Without A Metatype Literal`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(completedConversion: converter)
          var text: String = ""
        }
        """
      } diagnostics: {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(completedConversion: converter)
                                               ┬────────
                                               ╰─ 🛑 @EdgeToolsGuide(completedConversion:) requires a strategy type followed by .self.
          var text: String = ""
        }
        """
      }
    }

    @Test
    func `Diagnoses A Converted Property Without A Default`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(completedConversion: TextConversion.self)
          var text: String
        }
        """
      } diagnostics: {
        """
        @EdgeToolsGenerable
        ┬──────────────────
        ╰─ 🛑 The nonoptional converted stream field 'text' requires a defaultValue for init(orInitial:).
        struct Message {
          @EdgeToolsGuide(completedConversion: TextConversion.self)
          var text: String
        }
        """
      }
    }

    @Test
    func `Diagnoses Capacity On A Converted Property`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(initialCapacity: 32, completedConversion: TextConversion.self)
          var text: String = ""
        }
        """
      } diagnostics: {
        """
        @EdgeToolsGenerable
        ┬──────────────────
        ╰─ 🛑 The stream field 'text' cannot combine initialCapacity with completedConversion.
        struct Message {
          @EdgeToolsGuide(initialCapacity: 32, completedConversion: TextConversion.self)
          var text: String = ""
        }
        """
      }
    }

    @Test
    func `Diagnoses String Storage On A Converted Property`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct Message {
          @EdgeToolsGuide(partialStrings: .string, completedConversion: TextConversion.self)
          var text: String = ""
        }
        """
      } diagnostics: {
        """
        @EdgeToolsGenerable
        ┬──────────────────
        ╰─ 🛑 The stream field 'text' cannot combine String storage with completedConversion.
        struct Message {
          @EdgeToolsGuide(partialStrings: .string, completedConversion: TextConversion.self)
          var text: String = ""
        }
        """
      }
    }

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
                                            ┬─────
                                            ╰─ 🛑 @EdgeToolsGenerable(partialStrings:) requires .streamString or .string.
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
                                          ┬─────
                                          ╰─ 🛑 @EdgeToolsGuide(partialStrings:) requires .streamString or .string.
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
                                           ┬─
                                           ╰─ 🛑 @EdgeToolsGuide(initialCapacity:) requires a nonnegative integer literal.
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
