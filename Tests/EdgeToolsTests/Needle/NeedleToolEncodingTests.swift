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

    let encodedObject = try JSONSerialization.jsonObject(with: tool.needlePromptEncoded())
    let expectedObject = try JSONSerialization.jsonObject(with: JSONEncoder().encode(tool))
    let encoded = try JSONSerialization.data(withJSONObject: encodedObject, options: .sortedKeys)
    let expected = try JSONSerialization.data(withJSONObject: expectedObject, options: .sortedKeys)

    expectNoDifference(encoded, expected)
  }
}
