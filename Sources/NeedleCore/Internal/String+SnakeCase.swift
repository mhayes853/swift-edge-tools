extension String {
  func snakeCased() -> String {
    guard !self.isEmpty else { return self }

    var words = [Range<String.Index>]()
    var wordStart = self.startIndex
    var searchRange = wordStart..<self.endIndex

    while let upperCaseRange = self[searchRange]
      .rangeOfCharacter(from: .uppercaseLetters, options: [])
    {
      let untilUpperCase = wordStart..<upperCaseRange.lowerBound
      words.append(untilUpperCase)

      searchRange = upperCaseRange.lowerBound..<searchRange.upperBound
      guard
        let lowerCaseRange = self[searchRange]
          .rangeOfCharacter(from: .lowercaseLetters, options: [])
      else {
        wordStart = searchRange.lowerBound
        break
      }

      let nextCharacterAfterCapital = self.index(after: upperCaseRange.lowerBound)
      if lowerCaseRange.lowerBound == nextCharacterAfterCapital {
        wordStart = upperCaseRange.lowerBound
      } else {
        let beforeLowerIndex = self.index(before: lowerCaseRange.lowerBound)
        words.append(upperCaseRange.lowerBound..<beforeLowerIndex)

        wordStart = beforeLowerIndex
      }
      searchRange = lowerCaseRange.upperBound..<searchRange.upperBound
    }
    words.append(wordStart..<searchRange.upperBound)
    let snakeCased = words.map { self[$0].lowercased() }.joined(separator: "_")
    return snakeCased.starts(with: "_") ? String(snakeCased.dropFirst()) : snakeCased
  }
}
