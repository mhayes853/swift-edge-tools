import CustomDump
import EdgeTools
import Foundation
import Testing

@Suite
struct `NeedlePrompt tests` {
  @Test
  func `Formats Properly`() throws {
    let prompt = NeedlePrompt(
      system: "You are a helpful assistant who can send emails.",
      user: "Send an email to Henry."
    )

    expectNoDifference(
      try prompt.formatted(tools: [.sendEmail]),
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
    let prompt = NeedlePrompt(system: "", user: "Weather?")

    let rendered = try prompt.formatted(tools: [.getWeather])
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
      ),
      includesSchemaInInstructions: true
    )

    let prompt = NeedlePrompt(system: "", user: "")
    let formatted = try prompt.formatted(tools: [tool])

    expectNoDifference(formatted.contains("\"name\":\"\(expectedName)\""), true)
  }

  @Test
  func `Omits Schemas Known By The Model`() throws {
    let prompt = NeedlePrompt(system: "", user: "")
    var innate = EdgeToolDefinition.sendEmail
    innate.includesSchemaInInstructions = false

    let formatted = try prompt.formatted(tools: [innate, .getWeather])

    expectNoDifference(formatted.contains(#""name":"send_email""#), false)
    expectNoDifference(formatted.contains(#""name":"get_weather""#), true)
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
      ),
      includesSchemaInInstructions: true
    )

    let prompt = NeedlePrompt(system: "", user: "")
    let formatted = try prompt.formatted(tools: [tool])

    expectNoDifference(formatted.contains(#""name":"say_\"hello\""#), true)
    expectNoDifference(formatted.contains(#""description":"Uses a \"quoted\" phrase"#), true)
    expectNoDifference(formatted.contains(#""pattern":"say \\\"hi\\\""#), true)
  }
}
