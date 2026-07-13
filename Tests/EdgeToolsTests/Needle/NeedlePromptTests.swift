import CustomDump
import EdgeTools
import Testing

@Suite
struct `EdgeToolsPrompt tests` {
  @Test
  func `Formats Properly`() throws {
    let prompt = EdgeToolsPrompt(
      system: "You are a helpful assistant who can send emails.",
      user: "Send an email to Henry.",
      tools: [.sendEmail]
    )

    expectNoDifference(
      try prompt.needleFormatted(),
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
    let prompt = EdgeToolsPrompt(
      system: "",
      user: "Weather?",
      tools: [.getWeather]
    )

    let rendered = try prompt.needleFormatted()
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
}
