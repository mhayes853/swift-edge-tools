import EdgeTools

// MARK: - Default Prompts

extension NeedlePrompt {
  static let sendAdventureEmailTools = [SendEmailTool()]
  static let sendAdventureEmailDefinitions = sendAdventureEmailTools.map(\.definition)

  static let sendAdventureEmail = Self(
    system: "",
    user: "Send an email to Henry asking him to go on an adventure."
  )
}

// MARK: - DefinitionTool

struct DefinitionTool: EdgeTool {
  typealias Input = String
  typealias Output = String

  let definition: EdgeToolDefinition

  init(_ definition: EdgeToolDefinition) {
    self.definition = definition
  }

  var name: String { self.definition.name }
  var description: String { self.definition.description }
  var arguments: EdgeToolsGenerationSchema { self.definition.arguments }
  var includesSchemaInInstructions: Bool { self.definition.includesSchemaInInstructions }

  func invoke(input: String) async throws -> sending String { "" }
}

// MARK: - SendEmailTool

struct SendEmailTool: EdgeTool {
  @EdgeToolsGenerable
  struct Input: Sendable {
    @EdgeToolsGuide(
      .pattern("[a-z][a-z0-9]{1,10}@gmail\\.com"),
      .description("The recipient's email address."),
      .examples(["blob@gmail.com"])
    )
    var address: String

    @EdgeToolsGuide(.description("The subject of an email."))
    var subject: String

    @EdgeToolsGuide(.description("The content of an email."))
    var body: String
  }

  let name = "sendEmail"
  let description = "Sends an email to a recipient with an email address."

  func invoke(input: Input) async throws -> String {
    "Sent email to \(input.address)"
  }
}
