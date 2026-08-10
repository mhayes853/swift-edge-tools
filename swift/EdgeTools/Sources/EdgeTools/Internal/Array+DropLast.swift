extension Array {
  func dropLast(while predicate: (Element) -> Bool) -> Self {
    var result = self
    while let last = result.last, predicate(last) {
      result.removeLast()
    }
    return result
  }
}
