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
                .required(["query"]),
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
              let payload = try _edgeToolsRequireObjectValue(value, keys: ["query"])
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
        #"""
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

        extension Person: EdgeToolsGenerable, StreamParsingCore.StreamParseable {
          struct Partial: StreamParsingCore.StreamParseable,
            StreamParsingCore.StreamParseableObject, Sendable, EdgeToolsGenerable {
            typealias Partial = Self


            var name: String.Partial?


            init(
                name: String.Partial? = nil
              ) {
                self.name = name
              }


            private static let _streamInitialValueTemplate: Self = Self()

            static func streamInitialValue() -> Self {
              Self._streamInitialValueTemplate
            }


            #if !hasFeature(Embedded)
              static var streamObservationFields: [PartialKeyPath<Self>] {
                [\.name]
              }
              #endif


            @unsafe struct View: ~Copyable {
                let _streamStorage: UnsafeMutablePointer<Partial>

                init(_ storage: UnsafeMutableRawPointer) {
                  self._streamStorage = storage.assumingMemoryBound(to: Partial.self)
                }

                var name: String.Partial.View? {
                  get {
                    guard let address = StreamParsingCore._streamMemberAddress(&self._streamStorage.pointee.name) else {
                      return nil
                    }
                    return String.Partial.streamView(address)
                  }
                }
              }


            @unsafe
              static func streamView(_ storage: UnsafeMutableRawPointer) -> View {
                View(storage)
              }


            private enum StreamField {
                static let name: Int32 = 0
              }


            private static let streamContainerSchema_name = _streamContainerSchema(for: (String.Partial).self)


            static func streamMatchField(_ key: Span<UInt8>) -> Int32 {
                switch key.paddedLeadingWord() {
                case 0x0000_0000_656D_616E where key.count == 4:
                return Self.StreamField.name
                default:
                return -1
                }
              }


            static func streamApplyString(
                _ storage: UnsafeMutableRawPointer, _ field: Int32,
                _ bytes: Span<UInt8>
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.name:
                return streamApply(&p.pointee.name, utf8: bytes)
                default:
                return .unsupported
                }
              }


            static func streamApplyNumber(
                _ storage: UnsafeMutableRawPointer, _ field: Int32,
                _ bytes: Span<UInt8>, _ info: StreamParsingCore.NumberInfo
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.name:
                return streamApply(&p.pointee.name, bytes: bytes, info: info)
                default:
                return .unsupported
                }
              }


            static func streamApplyBoolean(
                _ storage: UnsafeMutableRawPointer, _ field: Int32, _ value: Bool
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.name:
                return streamApply(&p.pointee.name, boolean: value)
                default:
                return .unsupported
                }
              }


            static func streamApplyNull(
                _ storage: UnsafeMutableRawPointer, _ field: Int32
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.name:
                return StreamParsing.streamApplyNull(&p.pointee.name)
                default:
                return .unsupported
                }
              }


            static let streamFields: [StreamParsingCore.StreamField] = StreamParsingCore._streamFields(
                of: Self.self, prototype: Self()
              ) { p in
                [
                  StreamParsingCore.StreamField(
                    key: "name", index: Self.StreamField.name,
                    route: _streamFieldRoute(&p.pointee.name, schema: Self.streamContainerSchema_name),
                    offset: StreamParsingCore._streamFieldOffset(&p.pointee.name, in: p)
                  ),
                ]
              }


            static let streamSchema = StreamParsingCore.StreamSchema(
                shape: .object,
                matchField: Self.streamMatchField,
                applyString: Self.streamApplyString,
                applyNumber: Self.streamApplyNumber,
                applyBoolean: Self.streamApplyBoolean,
                applyNull: Self.streamApplyNull,
                fields: Self.streamFields
              )


            static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
                EdgeToolsGenerationSchema(
                  .type(.object),
                        .properties([
                              "name": String.Partial?.edgeToolsGenerationSchema
                            ])
                )
              }
            init(edgeToolsValue: EdgeToolsValue) throws {
              let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
              self.name = try Optional<String.Partial>(edgeToolsValue: _edgeToolsValue(object, forKey: "name"))
            }
            var edgeToolsValue: EdgeToolsValue {
              _edgeToolsBuildObjectValue(
                (key: "name", value: self.name?.edgeToolsValue)
              )
            }

          }
          var streamPartialValue: Partial {
            Partial(
              name: self.name.streamPartialValue
            )
          }

          init?(_ partial: Partial) {
            self.init(streamPartial: partial)
          }

          init?(streamPartial partial: Partial) {
            guard
              let name = Self._streamValue({ $0.name
              }, partial.name)
            else {
              return nil
            }
            self.name = name
          }

          init(orInitial partial: Partial) {
            self.name = Self._streamValueOrInitial({
                $0.name
              }, partial.name)
          }

          static func streamValueOrInitial(from partial: Partial) -> Self {
            Self(orInitial: partial)
          }
        }
        """#
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
        #"""
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

        extension Person: EdgeToolsGenerable, StreamParsingCore.StreamParseable {
          struct Partial: StreamParsingCore.StreamParseable,
            StreamParsingCore.StreamParseableObject, Sendable, EdgeToolsGenerable {
            typealias Partial = Self


            var firstName: String.Partial?


            init(
                firstName: String.Partial? = nil
              ) {
                self.firstName = firstName
              }


            private static let _streamInitialValueTemplate: Self = Self()

            static func streamInitialValue() -> Self {
              Self._streamInitialValueTemplate
            }


            #if !hasFeature(Embedded)
              static var streamObservationFields: [PartialKeyPath<Self>] {
                [\.firstName]
              }
              #endif


            @unsafe struct View: ~Copyable {
                let _streamStorage: UnsafeMutablePointer<Partial>

                init(_ storage: UnsafeMutableRawPointer) {
                  self._streamStorage = storage.assumingMemoryBound(to: Partial.self)
                }

                var firstName: String.Partial.View? {
                  get {
                    guard let address = StreamParsingCore._streamMemberAddress(&self._streamStorage.pointee.firstName) else {
                      return nil
                    }
                    return String.Partial.streamView(address)
                  }
                }
              }


            @unsafe
              static func streamView(_ storage: UnsafeMutableRawPointer) -> View {
                View(storage)
              }


            private enum StreamField {
                static let firstName: Int32 = 0
              }


            private static let streamContainerSchema_firstName = _streamContainerSchema(for: (String.Partial).self)


            static func streamMatchField(_ key: Span<UInt8>) -> Int32 {
                switch key.paddedLeadingWord() {
                case 0x616E_5F74_7372_6966 where key.count == 10 && key.paddedWord(at: 8) == 0x0000_0000_0000_656D:
                return Self.StreamField.firstName
                default:
                return -1
                }
              }


            static func streamApplyString(
                _ storage: UnsafeMutableRawPointer, _ field: Int32,
                _ bytes: Span<UInt8>
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.firstName:
                return streamApply(&p.pointee.firstName, utf8: bytes)
                default:
                return .unsupported
                }
              }


            static func streamApplyNumber(
                _ storage: UnsafeMutableRawPointer, _ field: Int32,
                _ bytes: Span<UInt8>, _ info: StreamParsingCore.NumberInfo
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.firstName:
                return streamApply(&p.pointee.firstName, bytes: bytes, info: info)
                default:
                return .unsupported
                }
              }


            static func streamApplyBoolean(
                _ storage: UnsafeMutableRawPointer, _ field: Int32, _ value: Bool
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.firstName:
                return streamApply(&p.pointee.firstName, boolean: value)
                default:
                return .unsupported
                }
              }


            static func streamApplyNull(
                _ storage: UnsafeMutableRawPointer, _ field: Int32
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.firstName:
                return StreamParsing.streamApplyNull(&p.pointee.firstName)
                default:
                return .unsupported
                }
              }


            static let streamFields: [StreamParsingCore.StreamField] = StreamParsingCore._streamFields(
                of: Self.self, prototype: Self()
              ) { p in
                [
                  StreamParsingCore.StreamField(
                    key: "first_name", index: Self.StreamField.firstName,
                    route: _streamFieldRoute(&p.pointee.firstName, schema: Self.streamContainerSchema_firstName),
                    offset: StreamParsingCore._streamFieldOffset(&p.pointee.firstName, in: p)
                  ),
                ]
              }


            static let streamSchema = StreamParsingCore.StreamSchema(
                shape: .object,
                matchField: Self.streamMatchField,
                applyString: Self.streamApplyString,
                applyNumber: Self.streamApplyNumber,
                applyBoolean: Self.streamApplyBoolean,
                applyNull: Self.streamApplyNull,
                fields: Self.streamFields
              )


            static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
                EdgeToolsGenerationSchema(
                  .type(.object),
                        .properties([
                              "first_name": EdgeToolsGenerationSchema(String.Partial?.edgeToolsGenerationSchema, .minLength(1), .description("Given name"), .examples(["Ada"]))
                            ])
                )
              }
            init(edgeToolsValue: EdgeToolsValue) throws {
              let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
              self.firstName = try Optional<String.Partial>(edgeToolsValue: _edgeToolsValue(object, forKey: "first_name"))
            }
            var edgeToolsValue: EdgeToolsValue {
              _edgeToolsBuildObjectValue(
                (key: "first_name", value: self.firstName?.edgeToolsValue)
              )
            }

          }
          var streamPartialValue: Partial {
            Partial(
              firstName: self.firstName.streamPartialValue
            )
          }

          init?(_ partial: Partial) {
            self.init(streamPartial: partial)
          }

          init?(streamPartial partial: Partial) {
            guard
              let firstName = Self._streamValue({ $0.firstName
              }, partial.firstName)
            else {
              return nil
            }
            self.firstName = firstName
          }

          init(orInitial partial: Partial) {
            self.firstName = Self._streamValueOrInitial({
                $0.firstName
              }, partial.firstName)
          }

          static func streamValueOrInitial(from partial: Partial) -> Self {
            Self(orInitial: partial)
          }
        }
        """#
      }
    }

    @Test
    func `Recognizes Qualified Property Macros And Package Access`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        package struct Payload {
          @EdgeTools.EdgeToolsGuide(key: "renamed")
          package var value: String
          @EdgeTools.EdgeToolsIgnored
          package var cache: Int = 0
        }
        """
      } expansion: {
        #"""
        package struct Payload {
          @EdgeTools.EdgeToolsGuide(key: "renamed")
          package var value: String
          @EdgeTools.EdgeToolsIgnored
          package var cache: Int = 0

          package static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
            EdgeToolsGenerationSchema(
              .type(.object),
                    .properties([
                          "renamed": String.edgeToolsGenerationSchema
                        ]),
                    .required(["renamed"])
            )
          }

          package init(edgeToolsValue: EdgeToolsValue) throws {
            let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
            self.value = try String(edgeToolsValue: _edgeToolsValue(object, forKey: "renamed"))
          }

          package var edgeToolsValue: EdgeToolsValue {
            _edgeToolsBuildObjectValue(
              (key: "renamed", value: self.value.edgeToolsValue)
            )
          }
        }

        extension Payload: EdgeToolsGenerable, StreamParsingCore.StreamParseable {
          package struct Partial: StreamParsingCore.StreamParseable,
            StreamParsingCore.StreamParseableObject, Sendable, EdgeToolsGenerable {
            package typealias Partial = Self


            package var value: String.Partial?


            package init(
                value: String.Partial? = nil
              ) {
                self.value = value
              }


            @usableFromInline static let _streamInitialValueTemplate: Self = Self()

            @inlinable package static func streamInitialValue() -> Self {
              Self._streamInitialValueTemplate
            }


            #if !hasFeature(Embedded)
              package static var streamObservationFields: [PartialKeyPath<Self>] {
                [\.value]
              }
              #endif


            @unsafe @frozen package struct View: ~Copyable {
                package let _streamStorage: UnsafeMutablePointer<Partial>

                @inlinable package init(_ storage: UnsafeMutableRawPointer) {
                  self._streamStorage = storage.assumingMemoryBound(to: Partial.self)
                }

                @inlinable package var value: String.Partial.View? {
                  get {
                    guard let address = StreamParsingCore._streamMemberAddress(&self._streamStorage.pointee.value) else {
                      return nil
                    }
                    return String.Partial.streamView(address)
                  }
                }
              }


            @unsafe
              @inlinable package static func streamView(_ storage: UnsafeMutableRawPointer) -> View {
                View(storage)
              }


            @usableFromInline enum StreamField {
                @inlinable static var value: Int32 {
                0
              }
              }


            private static let streamContainerSchema_value = _streamContainerSchema(for: (String.Partial).self)


            @inlinable package static func streamMatchField(_ key: Span<UInt8>) -> Int32 {
                switch key.paddedLeadingWord() {
                case 0x0064_656D_616E_6572 where key.count == 7:
                return Self.StreamField.value
                default:
                return -1
                }
              }


            @inlinable package static func streamApplyString(
                _ storage: UnsafeMutableRawPointer, _ field: Int32,
                _ bytes: Span<UInt8>
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.value:
                return streamApply(&p.pointee.value, utf8: bytes)
                default:
                return .unsupported
                }
              }


            @inlinable package static func streamApplyNumber(
                _ storage: UnsafeMutableRawPointer, _ field: Int32,
                _ bytes: Span<UInt8>, _ info: StreamParsingCore.NumberInfo
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.value:
                return streamApply(&p.pointee.value, bytes: bytes, info: info)
                default:
                return .unsupported
                }
              }


            @inlinable package static func streamApplyBoolean(
                _ storage: UnsafeMutableRawPointer, _ field: Int32, _ value: Bool
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.value:
                return streamApply(&p.pointee.value, boolean: value)
                default:
                return .unsupported
                }
              }


            @inlinable package static func streamApplyNull(
                _ storage: UnsafeMutableRawPointer, _ field: Int32
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.value:
                return StreamParsing.streamApplyNull(&p.pointee.value)
                default:
                return .unsupported
                }
              }


            package static let streamFields: [StreamParsingCore.StreamField] = StreamParsingCore._streamFields(
                of: Self.self, prototype: Self()
              ) { p in
                [
                  StreamParsingCore.StreamField(
                    key: "renamed", index: Self.StreamField.value,
                    route: _streamFieldRoute(&p.pointee.value, schema: Self.streamContainerSchema_value),
                    offset: StreamParsingCore._streamFieldOffset(&p.pointee.value, in: p)
                  ),
                ]
              }


            package static let streamSchema = StreamParsingCore.StreamSchema(
                shape: .object,
                matchField: Self.streamMatchField,
                applyString: Self.streamApplyString,
                applyNumber: Self.streamApplyNumber,
                applyBoolean: Self.streamApplyBoolean,
                applyNull: Self.streamApplyNull,
                fields: Self.streamFields
              )


            package static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
                EdgeToolsGenerationSchema(
                  .type(.object),
                        .properties([
                              "renamed": String.Partial?.edgeToolsGenerationSchema
                            ])
                )
              }
            package init(edgeToolsValue: EdgeToolsValue) throws {
              let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
              self.value = try Optional<String.Partial>(edgeToolsValue: _edgeToolsValue(object, forKey: "renamed"))
            }
            package var edgeToolsValue: EdgeToolsValue {
              _edgeToolsBuildObjectValue(
                (key: "renamed", value: self.value?.edgeToolsValue)
              )
            }

          }
          package var streamPartialValue: Partial {
            Partial(
              value: self.value.streamPartialValue
            )
          }

          @inlinable package init?(_ partial: Partial) {
            self.init(streamPartial: partial)
          }

          package init?(streamPartial partial: Partial) {
            guard
              let value = Self._streamValue({ $0.value
              }, partial.value)
            else {
              return nil
            }
            self.value = value
          }

          package init(orInitial partial: Partial) {
            self.value = Self._streamValueOrInitial({
                $0.value
              }, partial.value)
          }

          @inlinable package static func streamValueOrInitial(from partial: Partial) -> Self {
            Self(orInitial: partial)
          }
        }
        """#
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
        #"""
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

        extension Person: EdgeToolsGenerable, StreamParsingCore.StreamParseable {
          struct Partial: StreamParsingCore.StreamParseable,
            StreamParsingCore.StreamParseableObject, Sendable, EdgeToolsGenerable {
            typealias Partial = Self


            var name: String.Partial?


            init(
                name: String.Partial? = nil
              ) {
                self.name = name
              }


            private static let _streamInitialValueTemplate: Self = Self()

            static func streamInitialValue() -> Self {
              Self._streamInitialValueTemplate
            }


            #if !hasFeature(Embedded)
              static var streamObservationFields: [PartialKeyPath<Self>] {
                [\.name]
              }
              #endif


            @unsafe struct View: ~Copyable {
                let _streamStorage: UnsafeMutablePointer<Partial>

                init(_ storage: UnsafeMutableRawPointer) {
                  self._streamStorage = storage.assumingMemoryBound(to: Partial.self)
                }

                var name: String.Partial.View? {
                  get {
                    guard let address = StreamParsingCore._streamMemberAddress(&self._streamStorage.pointee.name) else {
                      return nil
                    }
                    return String.Partial.streamView(address)
                  }
                }
              }


            @unsafe
              static func streamView(_ storage: UnsafeMutableRawPointer) -> View {
                View(storage)
              }


            private enum StreamField {
                static let name: Int32 = 0
              }


            private static let streamContainerSchema_name = _streamContainerSchema(for: (String.Partial).self)


            static func streamMatchField(_ key: Span<UInt8>) -> Int32 {
                switch key.paddedLeadingWord() {
                case 0x0000_0000_656D_616E where key.count == 4:
                return Self.StreamField.name
                default:
                return -1
                }
              }


            static func streamApplyString(
                _ storage: UnsafeMutableRawPointer, _ field: Int32,
                _ bytes: Span<UInt8>
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.name:
                return streamApply(&p.pointee.name, utf8: bytes)
                default:
                return .unsupported
                }
              }


            static func streamApplyNumber(
                _ storage: UnsafeMutableRawPointer, _ field: Int32,
                _ bytes: Span<UInt8>, _ info: StreamParsingCore.NumberInfo
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.name:
                return streamApply(&p.pointee.name, bytes: bytes, info: info)
                default:
                return .unsupported
                }
              }


            static func streamApplyBoolean(
                _ storage: UnsafeMutableRawPointer, _ field: Int32, _ value: Bool
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.name:
                return streamApply(&p.pointee.name, boolean: value)
                default:
                return .unsupported
                }
              }


            static func streamApplyNull(
                _ storage: UnsafeMutableRawPointer, _ field: Int32
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.name:
                return StreamParsing.streamApplyNull(&p.pointee.name)
                default:
                return .unsupported
                }
              }


            static let streamFields: [StreamParsingCore.StreamField] = StreamParsingCore._streamFields(
                of: Self.self, prototype: Self()
              ) { p in
                [
                  StreamParsingCore.StreamField(
                    key: "name", index: Self.StreamField.name,
                    route: _streamFieldRoute(&p.pointee.name, schema: Self.streamContainerSchema_name),
                    offset: StreamParsingCore._streamFieldOffset(&p.pointee.name, in: p)
                  ),
                ]
              }


            static let streamSchema = StreamParsingCore.StreamSchema(
                shape: .object,
                matchField: Self.streamMatchField,
                applyString: Self.streamApplyString,
                applyNumber: Self.streamApplyNumber,
                applyBoolean: Self.streamApplyBoolean,
                applyNull: Self.streamApplyNull,
                fields: Self.streamFields
              )


            static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
                EdgeToolsGenerationSchema(
                  .type(.object),
                        .properties([
                              "name": String.Partial?.edgeToolsGenerationSchema
                            ])
                )
              }
            init(edgeToolsValue: EdgeToolsValue) throws {
              let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
              self.name = try Optional<String.Partial>(edgeToolsValue: _edgeToolsValue(object, forKey: "name"))
            }
            var edgeToolsValue: EdgeToolsValue {
              _edgeToolsBuildObjectValue(
                (key: "name", value: self.name?.edgeToolsValue)
              )
            }

          }
          var streamPartialValue: Partial {
            Partial(
              name: self.name.streamPartialValue
            )
          }

          init?(_ partial: Partial) {
            self.init(streamPartial: partial)
          }

          init?(streamPartial partial: Partial) {
            guard
              let name = Self._streamValue({ $0.name
              }, partial.name)
            else {
              return nil
            }
            self.name = name
          }

          init(orInitial partial: Partial) {
            self.name = Self._streamValueOrInitial({
                $0.name
              }, partial.name)
          }

          static func streamValueOrInitial(from partial: Partial) -> Self {
            Self(orInitial: partial)
          }
        }
        """#
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
        #"""
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

        extension Payload: EdgeToolsGenerable, StreamParsingCore.StreamParseable {
          struct Partial: StreamParsingCore.StreamParseable,
            StreamParsingCore.StreamParseableObject, Sendable, EdgeToolsGenerable {
            typealias Partial = Self


            var title: String.Partial?


            init(
                title: String.Partial? = nil
              ) {
                self.title = title
              }


            private static let _streamInitialValueTemplate: Self = Self()

            static func streamInitialValue() -> Self {
              Self._streamInitialValueTemplate
            }


            #if !hasFeature(Embedded)
              static var streamObservationFields: [PartialKeyPath<Self>] {
                [\.title]
              }
              #endif


            @unsafe struct View: ~Copyable {
                let _streamStorage: UnsafeMutablePointer<Partial>

                init(_ storage: UnsafeMutableRawPointer) {
                  self._streamStorage = storage.assumingMemoryBound(to: Partial.self)
                }

                var title: String.Partial.View? {
                  get {
                    guard let address = StreamParsingCore._streamMemberAddress(&self._streamStorage.pointee.title) else {
                      return nil
                    }
                    return String.Partial.streamView(address)
                  }
                }
              }


            @unsafe
              static func streamView(_ storage: UnsafeMutableRawPointer) -> View {
                View(storage)
              }


            private enum StreamField {
                static let title: Int32 = 0
              }


            private static let streamContainerSchema_title = _streamContainerSchema(for: (String.Partial).self)


            static func streamMatchField(_ key: Span<UInt8>) -> Int32 {
                switch key.paddedLeadingWord() {
                case 0x0000_0065_6C74_6974 where key.count == 5:
                return Self.StreamField.title
                default:
                return -1
                }
              }


            static func streamApplyString(
                _ storage: UnsafeMutableRawPointer, _ field: Int32,
                _ bytes: Span<UInt8>
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.title:
                return streamApply(&p.pointee.title, utf8: bytes)
                default:
                return .unsupported
                }
              }


            static func streamApplyNumber(
                _ storage: UnsafeMutableRawPointer, _ field: Int32,
                _ bytes: Span<UInt8>, _ info: StreamParsingCore.NumberInfo
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.title:
                return streamApply(&p.pointee.title, bytes: bytes, info: info)
                default:
                return .unsupported
                }
              }


            static func streamApplyBoolean(
                _ storage: UnsafeMutableRawPointer, _ field: Int32, _ value: Bool
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.title:
                return streamApply(&p.pointee.title, boolean: value)
                default:
                return .unsupported
                }
              }


            static func streamApplyNull(
                _ storage: UnsafeMutableRawPointer, _ field: Int32
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.title:
                return StreamParsing.streamApplyNull(&p.pointee.title)
                default:
                return .unsupported
                }
              }


            static let streamFields: [StreamParsingCore.StreamField] = StreamParsingCore._streamFields(
                of: Self.self, prototype: Self()
              ) { p in
                [
                  StreamParsingCore.StreamField(
                    key: "title", index: Self.StreamField.title,
                    route: _streamFieldRoute(&p.pointee.title, schema: Self.streamContainerSchema_title),
                    offset: StreamParsingCore._streamFieldOffset(&p.pointee.title, in: p)
                  ),
                ]
              }


            static let streamSchema = StreamParsingCore.StreamSchema(
                shape: .object,
                matchField: Self.streamMatchField,
                applyString: Self.streamApplyString,
                applyNumber: Self.streamApplyNumber,
                applyBoolean: Self.streamApplyBoolean,
                applyNull: Self.streamApplyNull,
                fields: Self.streamFields
              )


            static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
                EdgeToolsGenerationSchema(
                  .type(.object),
                        .properties([
                              "title": String.Partial?.edgeToolsGenerationSchema
                            ])
                )
              }
            init(edgeToolsValue: EdgeToolsValue) throws {
              let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
              self.title = try Optional<String.Partial>(edgeToolsValue: _edgeToolsValue(object, forKey: "title"))
            }
            var edgeToolsValue: EdgeToolsValue {
              _edgeToolsBuildObjectValue(
                (key: "title", value: self.title?.edgeToolsValue)
              )
            }

          }
          var streamPartialValue: Partial {
            Partial(
              title: self.title.streamPartialValue
            )
          }

          init?(_ partial: Partial) {
            self.init(streamPartial: partial)
          }

          init?(streamPartial partial: Partial) {
            guard
              let title = Self._streamValue({ $0.title
              }, partial.title)
            else {
              return nil
            }
            self.title = title
          }

          init(orInitial partial: Partial) {
            self.title = Self._streamValueOrInitial({
                $0.title
              }, partial.title)
          }

          static func streamValueOrInitial(from partial: Partial) -> Self {
            Self(orInitial: partial)
          }
        }
        """#
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

        extension Payload: EdgeToolsGenerable, StreamParsingCore.StreamParseable {
          struct Partial: StreamParsingCore.StreamParseable,
            StreamParsingCore.StreamParseableObject, Sendable, EdgeToolsGenerable {
            typealias Partial = Self


            init(

              ) {

              }


            private static let _streamInitialValueTemplate: Self = Self()

            static func streamInitialValue() -> Self {
              Self._streamInitialValueTemplate
            }


            #if !hasFeature(Embedded)
              static var streamObservationFields: [PartialKeyPath<Self>] {
                []
              }
              #endif


            @unsafe struct View: ~Copyable {
                let _streamStorage: UnsafeMutablePointer<Partial>

                init(_ storage: UnsafeMutableRawPointer) {
                  self._streamStorage = storage.assumingMemoryBound(to: Partial.self)
                }


              }


            @unsafe
              static func streamView(_ storage: UnsafeMutableRawPointer) -> View {
                View(storage)
              }


            static func streamMatchField(_ key: Span<UInt8>) -> Int32 {
                switch key.paddedLeadingWord() {

                default:
                return -1
                }
              }


            static func streamApplyString(
                _ storage: UnsafeMutableRawPointer, _ field: Int32,
                _ bytes: Span<UInt8>
              ) -> StreamParsingCore.StreamApplyResult {
                switch field {

                default:
                return .unsupported
                }
              }


            static func streamApplyNumber(
                _ storage: UnsafeMutableRawPointer, _ field: Int32,
                _ bytes: Span<UInt8>, _ info: StreamParsingCore.NumberInfo
              ) -> StreamParsingCore.StreamApplyResult {
                switch field {

                default:
                return .unsupported
                }
              }


            static func streamApplyBoolean(
                _ storage: UnsafeMutableRawPointer, _ field: Int32, _ value: Bool
              ) -> StreamParsingCore.StreamApplyResult {
                switch field {

                default:
                return .unsupported
                }
              }


            static func streamApplyNull(
                _ storage: UnsafeMutableRawPointer, _ field: Int32
              ) -> StreamParsingCore.StreamApplyResult {
                switch field {

                default:
                return .unsupported
                }
              }


            static let streamFields: [StreamParsingCore.StreamField] = StreamParsingCore._streamFields(
                of: Self.self, prototype: Self()
              ) { p in
                [

                ]
              }


            static let streamSchema = StreamParsingCore.StreamSchema(
                shape: .object,
                matchField: Self.streamMatchField,
                applyString: Self.streamApplyString,
                applyNumber: Self.streamApplyNumber,
                applyBoolean: Self.streamApplyBoolean,
                applyNull: Self.streamApplyNull,
                fields: Self.streamFields
              )


            static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
                EdgeToolsGenerationSchema(
                  .type(.object)
                )
              }
            init(edgeToolsValue: EdgeToolsValue) throws {
              _ = try _edgeToolsRequireObjectValue(edgeToolsValue)
            }
            var edgeToolsValue: EdgeToolsValue {
              _edgeToolsBuildObjectValue()
            }

          }
          var streamPartialValue: Partial {
            Partial()
          }

          init?(_ partial: Partial) {
            self.init(streamPartial: partial)
          }

          init?(streamPartial partial: Partial) {
            self.internalID = nil
          }

          init(orInitial partial: Partial) {
            self.internalID = nil
          }

          static func streamValueOrInitial(from partial: Partial) -> Self {
            Self(orInitial: partial)
          }
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
        #"""
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

        extension Person: EdgeToolsGenerable, StreamParsingCore.StreamParseable {
          struct Partial: StreamParsingCore.StreamParseable,
            StreamParsingCore.StreamParseableObject, Sendable, EdgeToolsGenerable {
            typealias Partial = Self


            var address: Address.Partial?


            init(
                address: Address.Partial? = nil
              ) {
                self.address = address
              }


            private static let _streamInitialValueTemplate: Self = Self()

            static func streamInitialValue() -> Self {
              Self._streamInitialValueTemplate
            }


            #if !hasFeature(Embedded)
              static var streamObservationFields: [PartialKeyPath<Self>] {
                [\.address]
              }
              #endif


            @unsafe struct View: ~Copyable {
                let _streamStorage: UnsafeMutablePointer<Partial>

                init(_ storage: UnsafeMutableRawPointer) {
                  self._streamStorage = storage.assumingMemoryBound(to: Partial.self)
                }

                var address: Address.Partial.View? {
                  get {
                    guard let address = StreamParsingCore._streamMemberAddress(&self._streamStorage.pointee.address) else {
                      return nil
                    }
                    return Address.Partial.streamView(address)
                  }
                }
              }


            @unsafe
              static func streamView(_ storage: UnsafeMutableRawPointer) -> View {
                View(storage)
              }


            private enum StreamField {
                static let address: Int32 = 0
              }


            private static let streamContainerSchema_address = _streamContainerSchema(for: (Address.Partial).self)


            static func streamMatchField(_ key: Span<UInt8>) -> Int32 {
                switch key.paddedLeadingWord() {
                case 0x0073_7365_7264_6461 where key.count == 7:
                return Self.StreamField.address
                default:
                return -1
                }
              }


            static func streamApplyString(
                _ storage: UnsafeMutableRawPointer, _ field: Int32,
                _ bytes: Span<UInt8>
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.address:
                return streamApply(&p.pointee.address, utf8: bytes)
                default:
                return .unsupported
                }
              }


            static func streamApplyNumber(
                _ storage: UnsafeMutableRawPointer, _ field: Int32,
                _ bytes: Span<UInt8>, _ info: StreamParsingCore.NumberInfo
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.address:
                return streamApply(&p.pointee.address, bytes: bytes, info: info)
                default:
                return .unsupported
                }
              }


            static func streamApplyBoolean(
                _ storage: UnsafeMutableRawPointer, _ field: Int32, _ value: Bool
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.address:
                return streamApply(&p.pointee.address, boolean: value)
                default:
                return .unsupported
                }
              }


            static func streamApplyNull(
                _ storage: UnsafeMutableRawPointer, _ field: Int32
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.address:
                return StreamParsing.streamApplyNull(&p.pointee.address)
                default:
                return .unsupported
                }
              }


            static let streamFields: [StreamParsingCore.StreamField] = StreamParsingCore._streamFields(
                of: Self.self, prototype: Self()
              ) { p in
                [
                  StreamParsingCore.StreamField(
                    key: "address", index: Self.StreamField.address,
                    route: _streamFieldRoute(&p.pointee.address, schema: Self.streamContainerSchema_address),
                    offset: StreamParsingCore._streamFieldOffset(&p.pointee.address, in: p)
                  ),
                ]
              }


            static let streamSchema = StreamParsingCore.StreamSchema(
                shape: .object,
                matchField: Self.streamMatchField,
                applyString: Self.streamApplyString,
                applyNumber: Self.streamApplyNumber,
                applyBoolean: Self.streamApplyBoolean,
                applyNull: Self.streamApplyNull,
                fields: Self.streamFields
              )


            static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
                EdgeToolsGenerationSchema(
                  .type(.object),
                        .properties([
                              "address": Address.Partial?.edgeToolsGenerationSchema
                            ])
                )
              }
            init(edgeToolsValue: EdgeToolsValue) throws {
              let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
              self.address = try Optional<Address.Partial>(edgeToolsValue: _edgeToolsValue(object, forKey: "address"))
            }
            var edgeToolsValue: EdgeToolsValue {
              _edgeToolsBuildObjectValue(
                (key: "address", value: self.address?.edgeToolsValue)
              )
            }

          }
          var streamPartialValue: Partial {
            Partial(
              address: self.address.streamPartialValue
            )
          }

          init?(_ partial: Partial) {
            self.init(streamPartial: partial)
          }

          init?(streamPartial partial: Partial) {
            guard
              let address = Self._streamValue({ $0.address
              }, partial.address)
            else {
              return nil
            }
            self.address = address
          }

          init(orInitial partial: Partial) {
            self.address = Self._streamValueOrInitial({
                $0.address
              }, partial.address)
          }

          static func streamValueOrInitial(from partial: Partial) -> Self {
            Self(orInitial: partial)
          }
        }
        """#
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
        #"""
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

        extension Person: EdgeToolsGenerable, StreamParsingCore.StreamParseable {
          struct Partial: StreamParsingCore.StreamParseable,
            StreamParsingCore.StreamParseableObject, Sendable, EdgeToolsGenerable {
            typealias Partial = Self


            var addresses: [Address].Partial?


            init(
                addresses: [Address].Partial? = nil
              ) {
                self.addresses = addresses
              }


            private static let _streamInitialValueTemplate: Self = Self()

            static func streamInitialValue() -> Self {
              Self._streamInitialValueTemplate
            }


            #if !hasFeature(Embedded)
              static var streamObservationFields: [PartialKeyPath<Self>] {
                [\.addresses]
              }
              #endif


            @unsafe struct View: ~Copyable {
                let _streamStorage: UnsafeMutablePointer<Partial>

                init(_ storage: UnsafeMutableRawPointer) {
                  self._streamStorage = storage.assumingMemoryBound(to: Partial.self)
                }

                var addresses: [Address].Partial.View? {
                  get {
                    guard let address = StreamParsingCore._streamMemberAddress(&self._streamStorage.pointee.addresses) else {
                      return nil
                    }
                    return [Address].Partial.streamView(address)
                  }
                }
              }


            @unsafe
              static func streamView(_ storage: UnsafeMutableRawPointer) -> View {
                View(storage)
              }


            private enum StreamField {
                static let addresses: Int32 = 0
              }


            private static let streamContainerSchema_addresses = _streamArraySchema(Address.Partial.self, element: _streamSchema(for: Address.Partial.self))


            static func streamMatchField(_ key: Span<UInt8>) -> Int32 {
                switch key.paddedLeadingWord() {
                case 0x6573_7365_7264_6461 where key.count == 9 && key.paddedWord(at: 8) == 0x0000_0000_0000_0073:
                return Self.StreamField.addresses
                default:
                return -1
                }
              }


            static func streamApplyString(
                _ storage: UnsafeMutableRawPointer, _ field: Int32,
                _ bytes: Span<UInt8>
              ) -> StreamParsingCore.StreamApplyResult {
                switch field {

                default:
                return .unsupported
                }
              }


            static func streamApplyNumber(
                _ storage: UnsafeMutableRawPointer, _ field: Int32,
                _ bytes: Span<UInt8>, _ info: StreamParsingCore.NumberInfo
              ) -> StreamParsingCore.StreamApplyResult {
                switch field {

                default:
                return .unsupported
                }
              }


            static func streamApplyBoolean(
                _ storage: UnsafeMutableRawPointer, _ field: Int32, _ value: Bool
              ) -> StreamParsingCore.StreamApplyResult {
                switch field {

                default:
                return .unsupported
                }
              }


            static func streamApplyNull(
                _ storage: UnsafeMutableRawPointer, _ field: Int32
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.addresses:
                return StreamParsing.streamApplyNull(&p.pointee.addresses)
                default:
                return .unsupported
                }
              }


            static let streamFields: [StreamParsingCore.StreamField] = StreamParsingCore._streamFields(
                of: Self.self, prototype: Self()
              ) { p in
                [
                  StreamParsingCore.StreamField(
                    key: "addresses", index: Self.StreamField.addresses,
                    route: _streamFieldRoute(&p.pointee.addresses, schema: Self.streamContainerSchema_addresses),
                    offset: StreamParsingCore._streamFieldOffset(&p.pointee.addresses, in: p)
                  ),
                ]
              }


            static let streamSchema = StreamParsingCore.StreamSchema(
                shape: .object,
                matchField: Self.streamMatchField,
                applyString: Self.streamApplyString,
                applyNumber: Self.streamApplyNumber,
                applyBoolean: Self.streamApplyBoolean,
                applyNull: Self.streamApplyNull,
                fields: Self.streamFields
              )


            static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
                EdgeToolsGenerationSchema(
                  .type(.object),
                        .properties([
                              "addresses": [Address].Partial?.edgeToolsGenerationSchema
                            ])
                )
              }
            init(edgeToolsValue: EdgeToolsValue) throws {
              let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
              self.addresses = try Optional<[Address].Partial>(edgeToolsValue: _edgeToolsValue(object, forKey: "addresses"))
            }
            var edgeToolsValue: EdgeToolsValue {
              _edgeToolsBuildObjectValue(
                (key: "addresses", value: self.addresses?.edgeToolsValue)
              )
            }

          }
          var streamPartialValue: Partial {
            Partial(
              addresses: self.addresses.streamPartialValue
            )
          }

          init?(_ partial: Partial) {
            self.init(streamPartial: partial)
          }

          init?(streamPartial partial: Partial) {
            guard
              let addresses = Self._streamValue({ $0.addresses
              }, partial.addresses)
            else {
              return nil
            }
            self.addresses = addresses
          }

          init(orInitial partial: Partial) {
            self.addresses = Self._streamValueOrInitial({
                $0.addresses
              }, partial.addresses)
          }

          static func streamValueOrInitial(from partial: Partial) -> Self {
            Self(orInitial: partial)
          }
        }
        """#
      }
    }
  }
}
