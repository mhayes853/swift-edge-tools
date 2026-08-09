// MARK: - DelimitedGenerationParser

struct DelimitedGenerationParser: Sendable {
  struct IgnoredToolRegion: Sendable {
    let opener: String
    let closer: String
  }

  private enum Region {
    case text
    case reasoning
    case tool
  }

  private let toolOpener: String
  private let toolCloser: String
  private let reasoningOpener: String?
  private let reasoningCloser: String?
  private let ignoredToolRegions: [IgnoredToolRegion]
  private let parseToolCalls: @Sendable (String) -> [EdgeRawToolCall]
  private var region: Region
  private var buffer = ""
  private var pendingParts = [EdgeToolsGenerationPart]()

  init(
    toolOpener: String,
    toolCloser: String,
    reasoningOpener: String? = nil,
    reasoningCloser: String? = nil,
    ignoredToolRegions: [IgnoredToolRegion] = [],
    parseToolCalls: @escaping @Sendable (String) -> [EdgeRawToolCall],
    startsInReasoning: Bool = false
  ) {
    self.toolOpener = toolOpener
    self.toolCloser = toolCloser
    self.reasoningOpener = reasoningOpener
    self.reasoningCloser = reasoningCloser
    self.ignoredToolRegions = ignoredToolRegions
    self.parseToolCalls = parseToolCalls
    self.region = startsInReasoning ? .reasoning : .text
  }

  mutating func accept(token: EdgeToolsToken) -> [EdgeToolsGenerationPart] {
    self.buffer.append(token.stringValue)
    return self.parseAvailableParts()
  }

  mutating func finish() -> [EdgeToolsGenerationPart] {
    var parts = self.parseAvailableParts()
    guard !self.buffer.isEmpty else { return parts }
    switch self.region {
    case .text:
      parts.append(.text(self.buffer))
    case .reasoning:
      parts.append(.reasoning(self.buffer))
    case .tool:
      parts.append(.text(self.toolOpener + self.buffer))
    }
    self.buffer = ""
    return parts
  }

  private mutating func parseAvailableParts() -> [EdgeToolsGenerationPart] {
    var parts = [EdgeToolsGenerationPart]()
    while let part = self.nextPart() {
      parts.append(part)
    }
    return parts
  }

  private mutating func nextPart() -> EdgeToolsGenerationPart? {
    if !self.pendingParts.isEmpty {
      return self.pendingParts.removeFirst()
    }
    switch self.region {
    case .text:
      return self.nextTextPart()
    case .reasoning:
      return self.nextReasoningPart()
    case .tool:
      return self.nextToolPart()
    }
  }

  private mutating func nextTextPart() -> EdgeToolsGenerationPart? {
    let markers = [self.reasoningOpener, self.toolOpener].compactMap { $0 }.filter { !$0.isEmpty }
    guard !markers.isEmpty else {
      guard !self.buffer.isEmpty else { return nil }
      defer { self.buffer = "" }
      return .text(self.buffer)
    }
    guard let match = self.firstMarker(in: self.buffer, markers: markers) else {
      return self.removeTextPrefix(retainingMarkerPrefix: markers)
    }
    guard !match.range.isEmpty else { return nil }
    if match.range.lowerBound > self.buffer.startIndex {
      let text = String(self.buffer[..<match.range.lowerBound])
      self.buffer.removeSubrange(..<match.range.lowerBound)
      return .text(text)
    }
    self.buffer.removeSubrange(..<match.range.upperBound)
    self.region = match.marker == self.reasoningOpener ? .reasoning : .tool
    return self.nextPart()
  }

  private mutating func nextReasoningPart() -> EdgeToolsGenerationPart? {
    guard let reasoningCloser else {
      guard !self.buffer.isEmpty else { return nil }
      defer { self.buffer = "" }
      return .reasoning(self.buffer)
    }
    guard let range = self.buffer.range(of: reasoningCloser) else {
      return self.removeReasoningPrefix(retainingCloserPrefix: reasoningCloser)
    }
    if range.lowerBound > self.buffer.startIndex {
      let reasoning = String(self.buffer[..<range.lowerBound])
      self.buffer.removeSubrange(..<range.lowerBound)
      return .reasoning(reasoning)
    }
    self.buffer.removeSubrange(..<range.upperBound)
    self.region = .text
    return self.nextPart()
  }

  private mutating func nextToolPart() -> EdgeToolsGenerationPart? {
    guard let range = self.toolCloserRange(in: self.buffer) else { return nil }
    let payload = String(self.buffer[..<range.lowerBound])
    self.buffer.removeSubrange(..<range.upperBound)
    self.region = .text
    let source = self.toolOpener + payload + self.toolCloser
    let calls = self.parseToolCalls(source)
    guard let call = calls.first else {
      return .text(source)
    }
    self.pendingParts.append(contentsOf: calls.dropFirst().map(EdgeToolsGenerationPart.toolCall))
    return .toolCall(call)
  }

  private func firstMarker(
    in source: String,
    markers: [String]
  ) -> (marker: String, range: Range<String.Index>)? {
    markers.compactMap { marker in
      source.range(of: marker).map { (marker, $0) }
    }
    .min { $0.range.lowerBound < $1.range.lowerBound }
  }

  private mutating func removeTextPrefix(
    retainingMarkerPrefix markers: [String]
  ) -> EdgeToolsGenerationPart? {
    let retainedCount = markers.map { self.buffer.suffixPrefixLength(of: $0) }.max() ?? 0
    guard self.buffer.count > retainedCount else { return nil }
    let end = self.buffer.index(self.buffer.endIndex, offsetBy: -retainedCount)
    let text = String(self.buffer[..<end])
    self.buffer.removeSubrange(..<end)
    return .text(text)
  }

  private mutating func removeReasoningPrefix(
    retainingCloserPrefix closer: String
  ) -> EdgeToolsGenerationPart? {
    let retainedCount = self.buffer.suffixPrefixLength(of: closer)
    guard self.buffer.count > retainedCount else { return nil }
    let end = self.buffer.index(self.buffer.endIndex, offsetBy: -retainedCount)
    let reasoning = String(self.buffer[..<end])
    self.buffer.removeSubrange(..<end)
    return .reasoning(reasoning)
  }

  private func toolCloserRange(in source: String) -> Range<String.Index>? {
    var cursor = source.startIndex
    while cursor < source.endIndex {
      let nextIgnoredRegion = self.ignoredToolRegions
        .compactMap { region in
          source.range(of: region.opener, range: cursor..<source.endIndex).map { (region, $0) }
        }
        .min { $0.1.lowerBound < $1.1.lowerBound }
      let toolCloser = source.range(of: self.toolCloser, range: cursor..<source.endIndex)
      guard let toolCloser else { return nil }
      guard let nextIgnoredRegion, nextIgnoredRegion.1.lowerBound < toolCloser.lowerBound else {
        return toolCloser
      }
      let afterOpener = nextIgnoredRegion.1.upperBound
      guard
        let ignoredCloser = source.range(
          of: nextIgnoredRegion.0.closer,
          range: afterOpener..<source.endIndex
        )
      else { return nil }
      cursor = ignoredCloser.upperBound
    }
    return nil
  }
}

extension String {
  func suffixPrefixLength(of marker: String) -> Int {
    let maximum = min(self.count, marker.count)
    guard maximum > 0 else { return 0 }
    return (1...maximum).reversed()
      .first { count in
        self.suffix(count) == marker.prefix(count)
      } ?? 0
  }
}
