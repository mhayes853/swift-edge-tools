extension String {
  func snakeCased() -> String {
    guard !self.isEmpty else { return self }

    var words = [Range<String.Index>]()
    var wordStart = self.startIndex
    var searchRange = wordStart..<self.endIndex

    while let upperCaseIndex = self[searchRange].firstIndex(where: \.isUppercase) {
      words.append(wordStart..<upperCaseIndex)

      searchRange = upperCaseIndex..<searchRange.upperBound
      guard let lowerCaseIndex = self[searchRange].firstIndex(where: \.isLowercase) else {
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
