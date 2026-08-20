enum EdgeToolsLLMInputKind: Equatable, Sendable {
  case generation
  case prefill
}

final class EdgeToolsLLMPreparedInputCache<Input: Sendable>: Sendable {
  #if FoundationEssentials
    private struct Entry: Sendable {
      let context: EdgeToolsLLMPrefillContext
      let kind: EdgeToolsLLMInputKind
      let input: Input
    }

    private let entry: Lock<Entry?>
  #endif

  init() {
    #if FoundationEssentials
      self.entry = Lock(nil)
    #endif
  }

  #if FoundationEssentials
    private init(entry: Entry?) {
      self.entry = Lock(entry)
    }

    func input(
      for context: EdgeToolsLLMPrefillContext,
      kind: EdgeToolsLLMInputKind,
      allowingTextOnlyContinuation: Bool = false
    ) -> Input? {
      self.entry.withBorrowedLock { entry in
        guard let entry else { return nil }
        if entry.context == context && entry.kind == kind {
          return entry.input
        }
        let continuation = entry.context.continuation(in: context)
        guard allowingTextOnlyContinuation, continuation == .textOnly else { return nil }
        return entry.input
      }
    }

    func store(
      _ input: Input,
      for context: EdgeToolsLLMPrefillContext,
      kind: EdgeToolsLLMInputKind
    ) {
      self.entry.withLock { entry in
        entry = Entry(context: context, kind: kind, input: input)
      }
    }

    func removeAll() {
      self.entry.withLock { entry in
        entry = nil
      }
    }
  #endif

  func input(
    for prompt: EdgeToolsTranscript,
    tools: [EdgeToolDefinition],
    kind: EdgeToolsLLMInputKind,
    allowingTextOnlyContinuation: Bool = false
  ) -> Input? {
    #if FoundationEssentials
      self.input(
        for: EdgeToolsLLMPrefillContext(prompt: prompt, tools: tools),
        kind: kind,
        allowingTextOnlyContinuation: allowingTextOnlyContinuation
      )
    #else
      nil
    #endif
  }

  func store(
    _ input: Input,
    for prompt: EdgeToolsTranscript,
    tools: [EdgeToolDefinition],
    kind: EdgeToolsLLMInputKind
  ) {
    #if FoundationEssentials
      self.store(
        input,
        for: EdgeToolsLLMPrefillContext(prompt: prompt, tools: tools),
        kind: kind
      )
    #endif
  }

  func forked() -> EdgeToolsLLMPreparedInputCache<Input> {
    #if FoundationEssentials
      EdgeToolsLLMPreparedInputCache(entry: self.entry.withBorrowedLock { $0 })
    #else
      EdgeToolsLLMPreparedInputCache()
    #endif
  }
}
