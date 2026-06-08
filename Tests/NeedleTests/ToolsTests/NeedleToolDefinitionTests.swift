import CustomDump
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
}
