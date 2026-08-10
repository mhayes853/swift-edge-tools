import EdgeToolsCLI

@main
struct EdgeMain {
  static func main() async {
    claimStandardOutput()
    await EdgeCommand.main()
  }
}
