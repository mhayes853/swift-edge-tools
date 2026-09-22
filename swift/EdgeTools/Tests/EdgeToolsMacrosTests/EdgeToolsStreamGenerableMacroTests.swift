import MacroTesting
import Testing

extension `EdgeToolsMacros tests` {
  @Suite
  struct `EdgeToolsStreamGenerableMacro tests` {
    @Test
    func `Generates Stream Parsing For A Struct Without An Argument`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct SearchRequest {
          @EdgeToolsGuide(key: "q")
          var query: String
        }
        """
      } expansion: {
        #"""
        struct SearchRequest {
          var query: String

          static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
            EdgeToolsGenerationSchema(
              .type(.object),
                    .properties([
                          "q": String.edgeToolsGenerationSchema
                        ]),
                    .required(["q"])
            )
          }

          init(edgeToolsValue: EdgeToolsValue) throws {
            let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
            self.query = try String(edgeToolsValue: _edgeToolsValue(object, forKey: "q"))
          }

          var edgeToolsValue: EdgeToolsValue {
            _edgeToolsBuildObjectValue(
              (key: "q", value: self.query.edgeToolsValue)
            )
          }
        }

        extension SearchRequest: EdgeToolsGenerable, StreamParsingCore.StreamParseable {
          struct Partial: StreamParsingCore.StreamParseable,
            StreamParsingCore.StreamParseableObject, Sendable, EdgeToolsGenerable {
            typealias Partial = Self


            var query: String.Partial?


            init(
                query: String.Partial? = nil
              ) {
                self.query = query
              }


            private static let _streamInitialValueTemplate: Self = Self()

            static func streamInitialValue() -> Self {
              Self._streamInitialValueTemplate
            }


            #if !hasFeature(Embedded)
              static var streamObservationFields: [PartialKeyPath<Self>] {
                [\.query]
              }
              #endif


            @unsafe struct View: ~Copyable {
                let _streamStorage: UnsafeMutablePointer<Partial>

                init(_ storage: UnsafeMutableRawPointer) {
                  self._streamStorage = storage.assumingMemoryBound(to: Partial.self)
                }

                var query: String.Partial.View? {
                  get {
                    guard let address = StreamParsingCore._streamMemberAddress(&self._streamStorage.pointee.query) else {
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
                static let query: Int32 = 0
              }


            private static let streamContainerSchema_query = _streamContainerSchema(for: (String.Partial).self)


            static func streamMatchField(_ key: Span<UInt8>) -> Int32 {
                switch key.paddedLeadingWord() {
                case 0x0000_0000_0000_0071 where key.count == 1:
                return Self.StreamField.query
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
                case Self.StreamField.query:
                return streamApply(&p.pointee.query, utf8: bytes)
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
                case Self.StreamField.query:
                return streamApply(&p.pointee.query, bytes: bytes, info: info)
                default:
                return .unsupported
                }
              }


            static func streamApplyBoolean(
                _ storage: UnsafeMutableRawPointer, _ field: Int32, _ value: Bool
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.query:
                return streamApply(&p.pointee.query, boolean: value)
                default:
                return .unsupported
                }
              }


            static func streamApplyNull(
                _ storage: UnsafeMutableRawPointer, _ field: Int32
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.query:
                return StreamParsing.streamApplyNull(&p.pointee.query)
                default:
                return .unsupported
                }
              }


            static let streamFields: [StreamParsingCore.StreamField] = StreamParsingCore._streamFields(
                of: Self.self, prototype: Self()
              ) { p in
                [
                  StreamParsingCore.StreamField(
                    key: "q", index: Self.StreamField.query,
                    route: _streamFieldRoute(&p.pointee.query, schema: Self.streamContainerSchema_query),
                    offset: StreamParsingCore._streamFieldOffset(&p.pointee.query, in: p)
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
                              "q": String.Partial?.edgeToolsGenerationSchema
                            ])
                )
              }
            init(edgeToolsValue: EdgeToolsValue) throws {
              let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
              self.query = try Optional<String.Partial>(edgeToolsValue: _edgeToolsValue(object, forKey: "q"))
            }
            var edgeToolsValue: EdgeToolsValue {
              _edgeToolsBuildObjectValue(
                (key: "q", value: self.query?.edgeToolsValue)
              )
            }

          }
          var streamPartialValue: Partial {
            Partial(
              query: self.query.streamPartialValue
            )
          }

          init?(_ partial: Partial) {
            self.init(streamPartial: partial)
          }

          init?(streamPartial partial: Partial) {
            guard
              let query = Self._streamValue({ $0.query
              }, partial.query)
            else {
              return nil
            }
            self.query = query
          }

          init(orInitial partial: Partial) {
            self.query = Self._streamValueOrInitial({
                $0.query
              }, partial.query)
          }

          static func streamValueOrInitial(from partial: Partial) -> Self {
            Self(orInitial: partial)
          }
        }
        """#
      }
    }

    @Test
    func `Generates Stream Parsing For A Defaulted Enum`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        enum Action {
          @StreamParseableDefault
          case idle(reason: String?)
          case search(query: String)
        }
        """
      } expansion: {
        #"""
        enum Action {
          @StreamParseableDefault
          case idle(reason: String?)
          case search(query: String)

          static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
            EdgeToolsGenerationSchema(
              .anyOf([
                      EdgeToolsGenerationSchema(
            .type(.object),
            .properties([
              "idle": EdgeToolsGenerationSchema(
                .type(.object),
                .properties([
                  "reason": String?.edgeToolsGenerationSchema
                ]),
                .additionalProperties(false)
              )
            ]),
            .required(["idle"]),
            .additionalProperties(false)
                            ),
                      EdgeToolsGenerationSchema(
            .type(.object),
            .properties([
              "search": EdgeToolsGenerationSchema(
                .type(.object),
                .properties([
                  "query": String.edgeToolsGenerationSchema
                ]),
                .required(["query"]),
                .additionalProperties(false)
              )
            ]),
            .required(["search"]),
            .additionalProperties(false)
                            )
                    ])
            )
          }

          init(edgeToolsValue: EdgeToolsValue) throws {
            let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
            if let value = object["idle"] {
            let payload = try _edgeToolsRequireObjectValue(value, keys: [])
            self = .idle(
              reason: try Optional<String>(edgeToolsValue: _edgeToolsValue(payload, forKey: "reason"))
            )
            return
            }
            if let value = object["search"] {
              let payload = try _edgeToolsRequireObjectValue(value, keys: ["query"])
              self = .search(
                query: try String(edgeToolsValue: _edgeToolsValue(payload, forKey: "query"))
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
            case .idle(let value0):
            _edgeToolsBuildObjectValue(
              (key: "idle", value: _edgeToolsBuildObjectValue(
                (key: "reason", value: value0.edgeToolsValue)
              )))
            case .search(let value0):
              _edgeToolsBuildObjectValue(
                (key: "search", value: _edgeToolsBuildObjectValue(
                  (key: "query", value: value0.edgeToolsValue)
                )))
            }
          }
        }

        extension Action: EdgeToolsGenerable, StreamParsingCore.StreamParseable {
          struct Partial: StreamParsingCore.StreamParseable,
            StreamParsingCore.StreamParseableObject, Sendable, EdgeToolsGenerable {
            typealias Partial = Self

            var idle: IdlePayload.Partial?
            var search: SearchPayload.Partial?

            init(
              idle: IdlePayload.Partial? = nil,
              search: SearchPayload.Partial? = nil
            ) {
              self.idle = idle
              self.search = search
            }

            private static let _streamInitialValueTemplate: Self = Self()

            static func streamInitialValue() -> Self {
              Self._streamInitialValueTemplate
            }

            #if !hasFeature(Embedded)
            static var streamObservationFields: [PartialKeyPath<Self>] {
              [\.idle, \.search]
            }
            #endif

            @unsafe struct View: ~Copyable {
              let _streamStorage: UnsafeMutablePointer<Partial>

              init(_ storage: UnsafeMutableRawPointer) {
                self._streamStorage = storage.assumingMemoryBound(to: Partial.self)
              }

              var idle: IdlePayload.Partial.View? {
                get {
                  guard let address = StreamParsingCore._streamMemberAddress(&self._streamStorage.pointee.idle) else {
                    return nil
                  }
                  return IdlePayload.Partial.streamView(address)
                }
              }

              var search: SearchPayload.Partial.View? {
                get {
                  guard let address = StreamParsingCore._streamMemberAddress(&self._streamStorage.pointee.search) else {
                    return nil
                  }
                  return SearchPayload.Partial.streamView(address)
                }
              }

              @unsafe enum ResolvedView: ~Copyable {
                case unresolved
                case ambiguous
                case idle(IdlePayload.Partial.View)
                case search(SearchPayload.Partial.View)
              }

              var resolved: ResolvedView {
                get {
                  var streamMatched = -1
                  var streamMatches = 0
                  if self._streamStorage.pointee.idle != nil {
                    streamMatched = 0
                    streamMatches += 1
                  }
                  if self._streamStorage.pointee.search != nil {
                    streamMatched = 1
                    streamMatches += 1
                  }
                  guard streamMatches == 1 else {
                    if streamMatches == 0 {
                      return .unresolved
                    }
                    return .ambiguous
                  }
                  switch streamMatched {
                  case 0:
                    guard let streamAddress = StreamParsingCore._streamMemberAddress(&self._streamStorage.pointee.idle) else {
                      return .unresolved
                    }
                    return .idle(IdlePayload.Partial.streamView(streamAddress))
                  case 1:
                    guard let streamAddress = StreamParsingCore._streamMemberAddress(&self._streamStorage.pointee.search) else {
                      return .unresolved
                    }
                    return .search(SearchPayload.Partial.streamView(streamAddress))
                  default:
                    return .unresolved
                  }
                }
              }
            }

            @unsafe
            static func streamView(_ storage: UnsafeMutableRawPointer) -> View {
              View(storage)
            }

            private enum StreamField {
              static let idle: Int32 = 0
              static let search: Int32 = 1
            }

            private static let streamContainerSchema_idle = _streamContainerSchema(for: (IdlePayload.Partial).self)
            private static let streamContainerSchema_search = _streamContainerSchema(for: (SearchPayload.Partial).self)

            static func streamMatchField(_ key: Span<UInt8>) -> Int32 {
              switch key.paddedLeadingWord() {
              case 0x0000_0000_656C_6469 where key.count == 4:
                return Self.StreamField.idle
              case 0x0000_6863_7261_6573 where key.count == 6:
                return Self.StreamField.search
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
              case Self.StreamField.idle:
                return streamApply(&p.pointee.idle, utf8: bytes)
              case Self.StreamField.search:
                return streamApply(&p.pointee.search, utf8: bytes)
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
              case Self.StreamField.idle:
                return streamApply(&p.pointee.idle, bytes: bytes, info: info)
              case Self.StreamField.search:
                return streamApply(&p.pointee.search, bytes: bytes, info: info)
              default:
                return .unsupported
              }
            }

            static func streamApplyBoolean(
              _ storage: UnsafeMutableRawPointer, _ field: Int32, _ value: Bool
            ) -> StreamParsingCore.StreamApplyResult {
              let p = storage.assumingMemoryBound(to: Self.self)
              switch field {
              case Self.StreamField.idle:
                return streamApply(&p.pointee.idle, boolean: value)
              case Self.StreamField.search:
                return streamApply(&p.pointee.search, boolean: value)
              default:
                return .unsupported
              }
            }

            static func streamApplyNull(
              _ storage: UnsafeMutableRawPointer, _ field: Int32
            ) -> StreamParsingCore.StreamApplyResult {
              let p = storage.assumingMemoryBound(to: Self.self)
              switch field {
              case Self.StreamField.idle:
                return StreamParsing.streamApplyNull(&p.pointee.idle)
              case Self.StreamField.search:
                return StreamParsing.streamApplyNull(&p.pointee.search)
              default:
                return .unsupported
              }
            }

            static let streamFields: [StreamParsingCore.StreamField] = StreamParsingCore._streamFields(
              of: Self.self, prototype: Self()
            ) { p in
              [
                StreamParsingCore.StreamField(
                  key: "idle", index: Self.StreamField.idle,
                  route: _streamFieldRoute(&p.pointee.idle, schema: Self.streamContainerSchema_idle),
                  offset: StreamParsingCore._streamFieldOffset(&p.pointee.idle, in: p)
                ),
                StreamParsingCore.StreamField(
                  key: "search", index: Self.StreamField.search,
                  route: _streamFieldRoute(&p.pointee.search, schema: Self.streamContainerSchema_search),
                  offset: StreamParsingCore._streamFieldOffset(&p.pointee.search, in: p)
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
                            "idle": IdlePayload.Partial?.edgeToolsGenerationSchema,
                      "search": SearchPayload.Partial?.edgeToolsGenerationSchema
                          ])
              )
            }
            init(edgeToolsValue: EdgeToolsValue) throws {
              let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
              self.idle = try Optional<IdlePayload.Partial>(edgeToolsValue: _edgeToolsValue(object, forKey: "idle"))
              self.search = try Optional<SearchPayload.Partial>(edgeToolsValue: _edgeToolsValue(object, forKey: "search"))
            }
            var edgeToolsValue: EdgeToolsValue {
              _edgeToolsBuildObjectValue(
                (key: "idle", value: self.idle?.edgeToolsValue),
                      (key: "search", value: self.search?.edgeToolsValue)
              )
            }

          }

          enum IdlePayload {
            struct Partial: StreamParsingCore.StreamParseable,
              StreamParsingCore.StreamParseableObject, Sendable, EdgeToolsGenerable {
              typealias Partial = Self

              var reason: String.Partial?

              init(
                reason: String.Partial? = nil
              ) {
                self.reason = reason
              }

              private static let _streamInitialValueTemplate: Self = Self()

              static func streamInitialValue() -> Self {
                Self._streamInitialValueTemplate
              }

              #if !hasFeature(Embedded)
              static var streamObservationFields: [PartialKeyPath<Self>] {
                [\.reason]
              }
              #endif

              @unsafe struct View: ~Copyable {
                let _streamStorage: UnsafeMutablePointer<Partial>

                init(_ storage: UnsafeMutableRawPointer) {
                  self._streamStorage = storage.assumingMemoryBound(to: Partial.self)
                }

                var reason: String.Partial.View? {
                  get {
                    guard let address = StreamParsingCore._streamMemberAddress(&self._streamStorage.pointee.reason) else {
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
                static let reason: Int32 = 0
              }

              private static let streamContainerSchema_reason = _streamContainerSchema(for: (String.Partial).self)

              static func streamMatchField(_ key: Span<UInt8>) -> Int32 {
                switch key.paddedLeadingWord() {
                case 0x0000_6E6F_7361_6572 where key.count == 6:
                  return Self.StreamField.reason
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
                case Self.StreamField.reason:
                  return streamApply(&p.pointee.reason, utf8: bytes)
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
                case Self.StreamField.reason:
                  return streamApply(&p.pointee.reason, bytes: bytes, info: info)
                default:
                  return .unsupported
                }
              }

              static func streamApplyBoolean(
                _ storage: UnsafeMutableRawPointer, _ field: Int32, _ value: Bool
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.reason:
                  return streamApply(&p.pointee.reason, boolean: value)
                default:
                  return .unsupported
                }
              }

              static func streamApplyNull(
                _ storage: UnsafeMutableRawPointer, _ field: Int32
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.reason:
                  return StreamParsing.streamApplyNull(&p.pointee.reason)
                default:
                  return .unsupported
                }
              }

              static let streamFields: [StreamParsingCore.StreamField] = StreamParsingCore._streamFields(
                of: Self.self, prototype: Self()
              ) { p in
                [
                  StreamParsingCore.StreamField(
                    key: "reason", index: Self.StreamField.reason,
                    route: _streamFieldRoute(&p.pointee.reason, schema: Self.streamContainerSchema_reason),
                    offset: StreamParsingCore._streamFieldOffset(&p.pointee.reason, in: p)
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
                              "reason": String.Partial?.edgeToolsGenerationSchema
                            ])
                )
              }
              init(edgeToolsValue: EdgeToolsValue) throws {
                let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
                self.reason = try Optional<String.Partial>(edgeToolsValue: _edgeToolsValue(object, forKey: "reason"))
              }
              var edgeToolsValue: EdgeToolsValue {
                _edgeToolsBuildObjectValue(
                  (key: "reason", value: self.reason?.edgeToolsValue)
                )
              }

            }

            struct Value: StreamParsingCore.StreamParseable {
              var reason: String?

              typealias Partial = IdlePayload.Partial

              var streamPartialValue: Partial {
                Partial(
                  reason: self.reason.streamPartialValue
                )
              }

              init?(_ partial: Partial) {
                self.init(streamPartial: partial)
              }

              init?(streamPartial partial: Partial) {
                guard
                  let reason = Self._streamValue({ $0.reason
                  }, partial.reason)
                else {
                  return nil
                }
                self.reason = reason
              }

              init(orInitial partial: Partial) {
                self.reason = Self._streamValueOrInitial({
                    $0.reason
                  }, partial.reason)
              }

              static func streamValueOrInitial(from partial: Partial) -> Self {
                Self(orInitial: partial)
              }
            }
          }

          enum SearchPayload {
            struct Partial: StreamParsingCore.StreamParseable,
              StreamParsingCore.StreamParseableObject, Sendable, EdgeToolsGenerable {
              typealias Partial = Self

              var query: String.Partial?

              init(
                query: String.Partial? = nil
              ) {
                self.query = query
              }

              private static let _streamInitialValueTemplate: Self = Self()

              static func streamInitialValue() -> Self {
                Self._streamInitialValueTemplate
              }

              #if !hasFeature(Embedded)
              static var streamObservationFields: [PartialKeyPath<Self>] {
                [\.query]
              }
              #endif

              @unsafe struct View: ~Copyable {
                let _streamStorage: UnsafeMutablePointer<Partial>

                init(_ storage: UnsafeMutableRawPointer) {
                  self._streamStorage = storage.assumingMemoryBound(to: Partial.self)
                }

                var query: String.Partial.View? {
                  get {
                    guard let address = StreamParsingCore._streamMemberAddress(&self._streamStorage.pointee.query) else {
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
                static let query: Int32 = 0
              }

              private static let streamContainerSchema_query = _streamContainerSchema(for: (String.Partial).self)

              static func streamMatchField(_ key: Span<UInt8>) -> Int32 {
                switch key.paddedLeadingWord() {
                case 0x0000_0079_7265_7571 where key.count == 5:
                  return Self.StreamField.query
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
                case Self.StreamField.query:
                  return streamApply(&p.pointee.query, utf8: bytes)
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
                case Self.StreamField.query:
                  return streamApply(&p.pointee.query, bytes: bytes, info: info)
                default:
                  return .unsupported
                }
              }

              static func streamApplyBoolean(
                _ storage: UnsafeMutableRawPointer, _ field: Int32, _ value: Bool
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.query:
                  return streamApply(&p.pointee.query, boolean: value)
                default:
                  return .unsupported
                }
              }

              static func streamApplyNull(
                _ storage: UnsafeMutableRawPointer, _ field: Int32
              ) -> StreamParsingCore.StreamApplyResult {
                let p = storage.assumingMemoryBound(to: Self.self)
                switch field {
                case Self.StreamField.query:
                  return StreamParsing.streamApplyNull(&p.pointee.query)
                default:
                  return .unsupported
                }
              }

              static let streamFields: [StreamParsingCore.StreamField] = StreamParsingCore._streamFields(
                of: Self.self, prototype: Self()
              ) { p in
                [
                  StreamParsingCore.StreamField(
                    key: "query", index: Self.StreamField.query,
                    route: _streamFieldRoute(&p.pointee.query, schema: Self.streamContainerSchema_query),
                    offset: StreamParsingCore._streamFieldOffset(&p.pointee.query, in: p)
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
                              "query": String.Partial?.edgeToolsGenerationSchema
                            ])
                )
              }
              init(edgeToolsValue: EdgeToolsValue) throws {
                let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
                self.query = try Optional<String.Partial>(edgeToolsValue: _edgeToolsValue(object, forKey: "query"))
              }
              var edgeToolsValue: EdgeToolsValue {
                _edgeToolsBuildObjectValue(
                  (key: "query", value: self.query?.edgeToolsValue)
                )
              }

            }

            struct Value: StreamParsingCore.StreamParseable {
              var query: String

              typealias Partial = SearchPayload.Partial

              var streamPartialValue: Partial {
                Partial(
                  query: self.query.streamPartialValue
                )
              }

              init?(_ partial: Partial) {
                self.init(streamPartial: partial)
              }

              init?(streamPartial partial: Partial) {
                guard
                  let query = Self._streamValue({ $0.query
                  }, partial.query)
                else {
                  return nil
                }
                self.query = query
              }

              init(orInitial partial: Partial) {
                self.query = Self._streamValueOrInitial({
                    $0.query
                  }, partial.query)
              }

              static func streamValueOrInitial(from partial: Partial) -> Self {
                Self(orInitial: partial)
              }
            }
          }
          var streamPartialValue: Partial {
            switch self {
            case .idle(let reason):
              return Partial(idle: IdlePayload.Partial(reason: reason.streamPartialValue))
            case .search(let query):
              return Partial(search: SearchPayload.Partial(query: query.streamPartialValue))
            }
          }

          init?(_ partial: Partial) {
            self.init(streamPartial: partial)
          }

          init?(streamPartial partial: Partial) {
            var streamMatched = -1
            var streamMatches = 0
            if partial.idle != nil {
              streamMatched = 0
              streamMatches += 1
            }
            if partial.search != nil {
              streamMatched = 1
              streamMatches += 1
            }
            guard streamMatches == 1 else {
              return nil
            }
            switch streamMatched {
            case 0:
              guard let streamValue = IdlePayload.Value(streamPartial: partial.idle!) else {
                return nil
              }
              self = .idle(reason: streamValue.reason)
            case 1:
              guard let streamValue = SearchPayload.Value(streamPartial: partial.search!) else {
                return nil
              }
              self = .search(query: streamValue.query)
            default:
              return nil
            }
          }

          init(orInitial partial: Partial) {
            if let streamMatched = Self(streamPartial: partial) {
              self = streamMatched
              return
            }
            let streamDefaultValue = IdlePayload.Value.streamValueOrInitial(
              from: partial.idle ?? IdlePayload.Partial.streamInitialValue()
            )
            self = .idle(reason: streamDefaultValue.reason)
          }

          static func streamValueOrInitial(from partial: Partial) -> Self {
            Self(orInitial: partial)
          }
        }
        """#
      }
    }

    @Test
    func `Preserves A User Declared Partial`() {
      assertMacro {
        """
        @EdgeToolsGenerable
        struct SearchRequest {
          var query: String
          struct Partial {}
        }
        """
      } expansion: {
        """
        struct SearchRequest {
          var query: String
          struct Partial {}

          static var edgeToolsGenerationSchema: EdgeToolsGenerationSchema {
            EdgeToolsGenerationSchema(
              .type(.object),
                    .properties([
                          "query": String.edgeToolsGenerationSchema
                        ]),
                    .required(["query"])
            )
          }

          init(edgeToolsValue: EdgeToolsValue) throws {
            let object = try _edgeToolsRequireObjectValue(edgeToolsValue)
            self.query = try String(edgeToolsValue: _edgeToolsValue(object, forKey: "query"))
          }

          var edgeToolsValue: EdgeToolsValue {
            _edgeToolsBuildObjectValue(
              (key: "query", value: self.query.edgeToolsValue)
            )
          }
        }

        extension SearchRequest: EdgeToolsGenerable {
        }
        """
      }
    }
  }
}
