import EdgeTools

// MARK: - Default Prompts

extension EdgeToolsPrompt {
  static let sendAdventureEmail = Self(
    system: "",
    user: "Send an email to Henry asking him to go on an adventure.",
    tools: [.sendEmail]
  )
}

// MARK: - SendEmailTool

struct SendEmailTool: EdgeTool {
  @EdgeToolsGenerable
  struct Input: Sendable {
    @EdgeToolsGuide(
      .pattern("[a-z][a-z0-9]{1,10}@gmail\\.com"),
      .description("The recipient's email address.")
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
