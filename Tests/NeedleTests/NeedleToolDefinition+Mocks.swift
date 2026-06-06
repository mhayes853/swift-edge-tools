import Needle

extension NeedleToolDefinition {
  static let sendEmail = Self(
    name: "send_email",
    description: "Sends an email to someone.",
    arguments: .object(
      properties: ["address": .string(), "subject": .string(), "body": .string()]
    )
  )
}
