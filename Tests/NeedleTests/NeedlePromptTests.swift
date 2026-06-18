import CustomDump
import Needle
import Testing

@Suite
struct `NeedlePrompt tests` {
  @Test
  func `Formats Properly`() throws {
    let prompt = NeedlePrompt(
      system: "You are a helpful assistant who can send emails.",
      user: "Send an email to Henry.",
      tools: [.sendEmail]
    )

    expectNoDifference(
      try prompt.formatted(),
      """
      You are a helpful assistant who can send emails.

      Send an email to Henry.<tools>[{"arguments":{"properties":{"address":{"description":"The recipient's email address.","pattern":"[a-z][a-z0-9]{1,10}@gmail\\\\.com","type":"string"},"body":{"type":"string"},"subject":{"type":"string"}},"required":["address","subject","body"],"type":"object"},"description":"Sends an email to a recipient with an email address.","name":"send_email"}]</s>
      """
    )
  }
}