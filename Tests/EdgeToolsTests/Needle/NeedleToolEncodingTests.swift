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
  func `Prompt Formatting Normalizes Tool Names`(name: String, expectedName: String) throws {
    let tool = EdgeToolDefinition(
      name: name,
      description: "Blob",
      arguments: EdgeToolsGenerationSchema(
        .type(.object),
        .properties(["name": .string])
      )
    )

    let prompt = EdgeToolsPrompt(system: "", user: "", tools: [tool])
    let formatted = try prompt.needleFormatted()

    expectNoDifference(formatted.contains("\"name\":\"\(expectedName)\""), true)
  }

  @Test
  func `Needle Prompt Formatting Escapes Quotes`() throws {
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

    let prompt = EdgeToolsPrompt(system: "", user: "", tools: [tool])
    let encoded = try prompt.needleFormatted()

    expectNoDifference(encoded.contains(#""name":"say_\"hello\""#), true)
    expectNoDifference(encoded.contains(#""description":"Uses a \"quoted\" phrase"#), true)
    expectNoDifference(encoded.contains(#""pattern":"say \\\"hi\\\""#), true)
  }
}
