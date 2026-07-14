import CustomDump
import Foundation
import EdgeTools
import Testing

@Suite
struct `NeedlePrompt tests` {
  @Test
  func `Formats Properly`() throws {
    let prompt = NeedlePrompt(
      system: "You are a helpful assistant who can send emails.",
      user: "Send an email to Henry.",
      tools: [DefinitionTool(.sendEmail)]
    )

    expectNoDifference(
      try prompt.formatted(),
      """
      You are a helpful assistant who can send emails.

      Send an email to Henry.<tools>[{"name":"send_email","description":"Sends an email to a recipient with an email address.",\
      "arguments":{"type":"object","properties":{"address":{"type":"string","description":"The recipient's email address.",\
      "pattern":"[a-z][a-z0-9]{1,10}@gmail\\\\.com","examples":["blob@gmail.com"]},"subject":{"type":"string"},\
      "body":{"type":"string"}},"required":["address","subject","body"]}}]
      """
    )
  }

  @Test
  func `Uses Canonical Tool And Schema Field Order`() throws {
    let prompt = NeedlePrompt(
      system: "",
      user: "Weather?",
      tools: [DefinitionTool(.getWeather)]
    )

    let rendered = try prompt.formatted()
    let jsonStart = rendered.firstIndex(of: "[") ?? rendered.endIndex
    let jsonSlice = rendered[jsonStart...]

    let expected =
      """
      [{"name":"get_weather","description":"Gets the current weather for a location.",\
      "arguments":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"],\
      "additionalProperties":false}}]
      """
    expectNoDifference(String(jsonSlice), expected)
  }

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
  func `Formats Normalized Tool Names`(name: String, expectedName: String) throws {
    let tool = EdgeToolDefinition(
      name: name,
      description: "Blob",
      arguments: EdgeToolsGenerationSchema(
        .type(.object),
        .properties(["name": .string])
      )
    )

    let prompt = NeedlePrompt(system: "", user: "", tools: [DefinitionTool(tool)])
    let formatted = try prompt.formatted()

    expectNoDifference(formatted.contains("\"name\":\"\(expectedName)\""), true)
  }

  @Test
  func `Formatting Escapes Quotes`() throws {
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

    let prompt = NeedlePrompt(system: "", user: "", tools: [DefinitionTool(tool)])
    let formatted = try prompt.formatted()

    expectNoDifference(formatted.contains(#""name":"say_\"hello\""#), true)
    expectNoDifference(formatted.contains(#""description":"Uses a \"quoted\" phrase"#), true)
    expectNoDifference(formatted.contains(#""pattern":"say \\\"hi\\\""#), true)
  }
}
