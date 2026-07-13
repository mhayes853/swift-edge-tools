import CustomDump
import Foundation
import EdgeTools
import Testing

@Suite
struct `Needle tool encoding tests` {
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
    let tool = EdgeToolDefinition(
      name: name,
      description: "Blob",
      arguments: EdgeToolsGenerationSchema(
        .type(.object),
        .properties(["name": .string])
      )
    )

    var expectedTool = tool
    expectedTool.name = expectedName
    expectNoDifference(tool.needleNormalized(), expectedTool)
  }

  @Test
  func `Needle Prompt Encoding Escapes Quotes`() throws {
    let tool = EdgeToolDefinition(
      name: "say_\"hello\"",
      description: "Uses a \"quoted\" phrase",
      arguments: EdgeToolsGenerationSchema(
        .type(.object),
        .description("Schema with \"quotes\""),
        .properties([
          "message": EdgeToolsGenerationSchema(
            .string,
            .pattern(#"say \"hi\""#)
          )
        ])
      )
    )

    let encoded = try tool.needlePromptEncoded()

    expectNoDifference(encoded.contains(#""name":"say_\"hello\""#), true)
    expectNoDifference(encoded.contains(#""description":"Uses a \"quoted\" phrase"#), true)
    expectNoDifference(encoded.contains(#""pattern":"say \\\"hi\\\""#), true)
  }
}
