#if XGrammar
  final class NeedleGrammarMatcherPool {
    private struct Key: Hashable, Sendable {
      let tools: [NeedleToolDefinition]
      let range: NeedleXGrammarEngine.ToolCallInvocationRange
    }

    private let maxCount: Int
    private var entries = [Key: NeedleXGrammarEngine.Matcher]()
    private var order = [Key]()

    init(maxCount: Int = 8) {
      self.maxCount = maxCount
    }

    func matcher(
      tools: some Sequence<NeedleToolDefinition>,
      range: NeedleXGrammarEngine.ToolCallInvocationRange,
      compilingWith engine: NeedleXGrammarEngine
    ) throws -> NeedleXGrammarEngine.Matcher {
      let key = Key(tools: tools.map { $0.normalized() }, range: range)
      if let cached = self.entries[key] {
        self.touch(key)
        return cached
      }
      let matcher = try engine.compile(tools: key.tools, range: key.range)
      self.insert(key, matcher)
      return matcher
    }

    func clear() {
      self.entries.removeAll()
      self.order.removeAll()
    }

    private func touch(_ key: Key) {
      self.order.removeAll { $0 == key }
      self.order.append(key)
    }

    private func insert(_ key: Key, _ matcher: NeedleXGrammarEngine.Matcher) {
      if self.entries.count >= self.maxCount, let leastRecentlyUsed = self.order.first {
        self.entries.removeValue(forKey: leastRecentlyUsed)
        self.order.removeFirst()
      }
      self.entries[key] = matcher
      self.order.append(key)
    }
  }
#endif
