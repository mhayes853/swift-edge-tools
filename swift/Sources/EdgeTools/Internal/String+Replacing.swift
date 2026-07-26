extension String {
  func replacing(_ target: String, with replacement: String) -> String {
    self.split(separator: target, omittingEmptySubsequences: false).joined(separator: replacement)
  }
}
