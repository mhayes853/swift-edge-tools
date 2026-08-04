// MARK: - Whitespace

extension StringProtocol {
  var trimmingWhitespace: String {
    String(
      self.drop { $0.isWhitespace }
        .reversed()
        .drop { $0.isWhitespace }
        .reversed()
    )
  }
}

// MARK: - Searching

extension StringProtocol {
  // NB: Embedded Swift's stdlib slice ships no firstRange(of:), so the search is hand-rolled.
  func firstRange(in source: Substring) -> Range<String.Index>? {
    guard !self.isEmpty else { return source.startIndex..<source.startIndex }
    for index in source.indices where source[index...].starts(with: self) {
      return index..<source.index(index, offsetBy: self.count)
    }
    return nil
  }
}

// MARK: - Replacing

extension String {
  func replacing(_ target: String, with replacement: String) -> String {
    guard !target.isEmpty else { return self }

    var result = ""
    var remainder = self[...]
    while let range = target.firstRange(in: remainder) {
      result += remainder[..<range.lowerBound]
      result += replacement
      remainder = remainder[range.upperBound...]
    }
    result += remainder
    return result
  }
}

// MARK: - Snake Case

extension String {
  func snakeCased() -> String {
    guard !self.isEmpty else { return self }

    var words = [Range<String.Index>]()
    var wordStart = self.startIndex
    var searchRange = wordStart..<self.endIndex

    while let upperCaseIndex = self[searchRange].firstIndex(where: { $0.isUppercase }) {
      words.append(wordStart..<upperCaseIndex)

      searchRange = upperCaseIndex..<searchRange.upperBound
      guard let lowerCaseIndex = self[searchRange].firstIndex(where: { $0.isLowercase }) else {
        wordStart = searchRange.lowerBound
        break
      }

      let nextCharacterAfterCapital = self.index(after: upperCaseIndex)
      if lowerCaseIndex == nextCharacterAfterCapital {
        wordStart = upperCaseIndex
      } else {
        let beforeLowerIndex = self.index(before: lowerCaseIndex)
        words.append(upperCaseIndex..<beforeLowerIndex)

        wordStart = beforeLowerIndex
      }
      searchRange = self.index(after: lowerCaseIndex)..<searchRange.upperBound
    }
    words.append(wordStart..<searchRange.upperBound)
    let snakeCased = words.map { self[$0].lowercased() }.joined(separator: "_")
    return snakeCased.starts(with: "_") ? String(snakeCased.dropFirst()) : snakeCased
  }
}
