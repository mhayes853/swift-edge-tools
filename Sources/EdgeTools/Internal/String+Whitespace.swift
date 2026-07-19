extension StringProtocol {
  var trimmingWhitespace: String {
    String(
      self.drop(while: \Character.isWhitespace)
        .reversed()
        .drop(while: \Character.isWhitespace)
        .reversed()
    )
  }
}
