#if XGrammar
  final class NeedleGrammarMatcherPool {
    private struct Key: Hashable, Sendable {
      let tools: [EdgeToolDefinition]
      let range: GrammarToolCallRange
    }

    private let maxCount: Int
    private var entries = [Key: XGrammarMatcher]()
    private var order = [Key]()

    init(maxCount: Int = 8) {
      self.maxCount = maxCount
    }

    func matcher(
      tools: some Sequence<EdgeToolDefinition>,
      range: GrammarToolCallRange,
      compilingWith compiler: XGrammarCompiler
    ) throws -> XGrammarMatcher {
      let key = Key(tools: tools.map { $0.needleNormalized() }, range: range)
      if let cached = self.entries[key] {
        self.touch(key)
        cached.reset()
        return cached.fork()
      }
      let grammar = try XGrammarGrammar.needle(tools: key.tools, range: key.range)
      let matcher = try compiler.compile(grammar)
      self.insert(key, matcher)
      return matcher.fork()
    }

    func clear() {
      self.entries.removeAll()
      self.order.removeAll()
    }

    private func touch(_ key: Key) {
      self.order.removeAll { $0 == key }
      self.order.append(key)
    }

    private func insert(_ key: Key, _ matcher: XGrammarMatcher) {
      if self.entries.count >= self.maxCount, let leastRecentlyUsed = self.order.first {
        self.entries.removeValue(forKey: leastRecentlyUsed)
        self.order.removeFirst()
      }
      self.entries[key] = matcher
      self.order.append(key)
    }
  }
#endif
