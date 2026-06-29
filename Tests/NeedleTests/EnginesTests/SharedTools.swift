import Needle

// MARK: - Default Prompts

extension NeedlePrompt {
  static let sendAdventureEmail = Self(
    system: "",
    user: "Send an email to Henry asking him to go on an adventure.",
    tools: [.sendEmail]
  )
}

// MARK: - SendEmailTool

struct SendEmailTool: NeedleTool {
  @NeedleGenerable
  struct Input: Sendable {
    @NeedleGuide(
      .string(pattern: /[a-z][a-z0-9]{1,10}@gmail\.com/),
      description: "The recipient's email address."
    )
    var address: String

    @NeedleGuide(description: "The subject of an email.")
    var subject: String

    @NeedleGuide(description: "The content of an email.")
    var body: String
  }

  let name = "sendEmail"
  let description = "Sends an email to a recipient with an email address."

  func invoke(input: Input) async throws -> String {
    "Sent email to \(input.address)"
  }
}
