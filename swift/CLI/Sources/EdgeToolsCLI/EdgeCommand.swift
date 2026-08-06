import ArgumentParser

// MARK: - EdgeCommand

public struct EdgeCommand: AsyncParsableCommand {
  public static let configuration = CommandConfiguration(
    commandName: "edge",
    abstract: "Run and benchmark on-device tool calling models.",
    subcommands: [RunCommand.self, BenchCommand.self, InfoCommand.self],
    defaultSubcommand: RunCommand.self
  )

  public init() {}
}
