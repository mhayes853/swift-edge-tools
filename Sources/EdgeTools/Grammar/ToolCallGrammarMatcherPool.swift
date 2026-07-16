#if XGrammar
  final class ToolCallGrammarMatcherPool {
    typealias NormalizeTools = @Sendable ([EdgeToolDefinition]) -> [EdgeToolDefinition]
    typealias MakeGrammar =
      @Sendable ([EdgeToolDefinition], GrammarToolCallRange) throws -> XGrammarGrammar

    private final class CachedMatcher {
      private let matcher: XGrammarMatcher

      init(_ matcher: consuming XGrammarMatcher) {
        self.matcher = consume matcher
      }

      func fork() throws -> XGrammarMatcher {
        try self.matcher.fork()
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
      compilingWith compiler: borrowing XGrammarCompiler
    ) throws -> XGrammarMatcher {
      let key = Key(tools: self.normalizeTools(Array(tools)), range: range)
      if let cached = self.entries[key] {
        self.touch(key)
        return try cached.fork()
      }
      let grammar = try self.makeGrammar(key.tools, key.range)
      let matcher = try compiler.compile(grammar)
      return try self.insert(key, matcher: consume matcher)
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
      matcher: consuming XGrammarMatcher
    ) throws -> XGrammarMatcher {
      let cached = CachedMatcher(consume matcher)
      if self.entries.count >= self.maxCount, let leastRecentlyUsed = self.order.first {
        self.entries.removeValue(forKey: leastRecentlyUsed)
        self.order.removeFirst()
      }
      self.entries[key] = cached
      self.order.append(key)
      return try cached.fork()
    }
  }
#endif
