import CustomDump
import Foundation
import Needle
import Testing

@Suite
struct `NeedleToolDefinition tests` {
  @Test(
    arguments: [
      ("sendEmail", "send_email"),
      ("sendEmailTo", "send_email_to"),
      ("SendEmailTo", "send_email_to"),
      ("send_email", "send_email"),
      ("", ""),
      ("sendEmail2", "send_email2"),
      ("send", "send"),
      ("Send", "send")
    ]
  )
  func `Normalized Basics`(name: String, expectedName: String) {
    let tool = NeedleToolDefinition(
      name: name,
      description: "Blob",
      arguments: .object(properties: ["name": .string()])
    )

    var expectedTool = tool
    expectedTool.name = expectedName
    expectNoDifference(tool.normalized(), expectedTool)
  }

  @Test
  func `Needle Prompt Encoding Escapes Quotes`() {
    let tool = NeedleToolDefinition(
      name: "say_\"hello\"",
      description: "Uses a \"quoted\" phrase",
      arguments: .object(
        description: "Schema with \"quotes\"",
        properties: ["message": .string(pattern: #"say \"hi\""#)]
      )
    )

    let decoded = try? JSONDecoder().decode(NeedleToolDefinition.self, from: tool.needlePromptEncoded())

    expectNoDifference(decoded, tool)
  }
}
