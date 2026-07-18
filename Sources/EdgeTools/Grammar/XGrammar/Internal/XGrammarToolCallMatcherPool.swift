#if XGrammar
  final class XGRToolCallMatcherPool {
    typealias NormalizeTools = ([EdgeToolDefinition]) -> [EdgeToolDefinition]
    typealias MakeGrammar =
      ([EdgeToolDefinition], GrammarToolCallRange) throws -> XGRGrammar

    private final class CachedMatcher {
      private let matcher: XGRMatcher

      init(_ matcher: consuming XGRMatcher) {
        self.matcher = consume matcher
      }

      func fork() -> XGRMatcher {
        self.matcher.fork()
      }
    }

    private struct Key: Hashable, Sendable {
      let tools: [EdgeToolDefinition]
      let range: GrammarToolCallRange
    }

    private let maxCount: Int
    private let normalizeTools: NormalizeTools
    private let makeGrammar: MakeGrammar
    private var entries = [Key: CachedMatcher]()
    private var order = [Key]()

    init(
      maxCount: Int = 8,
      normalizeTools: @escaping NormalizeTools = { $0 },
      makeGrammar: @escaping MakeGrammar
    ) {
      self.maxCount = maxCount
      self.normalizeTools = normalizeTools
      self.makeGrammar = makeGrammar
    }

    func matcher(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange,
      compilingWith compiler: borrowing XGRCompiler
    ) throws -> XGRMatcher {
      let key = Key(tools: self.normalizeTools(Array(tools)), range: range)
      if let cached = self.entries[key] {
        self.touch(key)
        return cached.fork()
      }
      let grammar = try self.makeGrammar(key.tools, key.range)
      let compiledGrammar = try compiler.compile(grammar)
      let matcher = try XGRMatcher(compiledGrammar: compiledGrammar)
      return self.insert(key, matcher: consume matcher)
    }

    func clear() {
      self.entries.removeAll()
      self.order.removeAll()
    }

    private func touch(_ key: Key) {
      self.order.removeAll { $0 == key }
      self.order.append(key)
    }

    private func insert(
      _ key: Key,
      matcher: consuming XGRMatcher
    ) -> XGRMatcher {
      let cached = CachedMatcher(consume matcher)
      if self.entries.count >= self.maxCount, let leastRecentlyUsed = self.order.first {
        self.entries.removeValue(forKey: leastRecentlyUsed)
        self.order.removeFirst()
      }
      self.entries[key] = cached
      self.order.append(key)
      return cached.fork()
    }
  }
#endif
