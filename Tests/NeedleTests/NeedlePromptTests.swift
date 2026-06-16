import CustomDump
import Needle
import Testing

@Suite
struct `NeedlePrompt tests` {
  @Test
  func `Formats Properly`() throws {
    var toolDefinition = NeedleToolDefinition.sendEmail
    toolDefinition.name = "sendEmail"

    let prompt = NeedlePrompt(
      system: "You are a helpful assistant who can send emails.",
      user: "Send an email to Henry.",
      tools: [.sendEmail]
    )

    expectNoDifference(
      try prompt.formatted(),
      """
      You are a helpful assistant who can send emails.

      Send an email to Henry.\
      <tools>[{\
      "arguments":{\
      "properties":{\
      "address":{"type":"string"},"body":{"type":"string"},"subject":{"type":"string"}},\
      "required":["address","subject","body"],\
      "type":"object"\
      },\
      "description":"Sends an email to someone.",\
      "name":"send_email"\
      }]</s>
      """
    )
  }
}
