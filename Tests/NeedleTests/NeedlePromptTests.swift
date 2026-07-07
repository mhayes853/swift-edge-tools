import CustomDump
import Needle
import Testing

@Suite
struct `NeedlePrompt tests` {
  @Test
  func `Formats Properly`() {
    let prompt = NeedlePrompt(
      system: "You are a helpful assistant who can send emails.",
      user: "Send an email to Henry.",
      tools: [.sendEmail]
    )

    expectNoDifference(
      prompt.formatted(),
      """
      You are a helpful assistant who can send emails.

      Send an email to Henry.<tools>[{"name":"send_email","description":"Sends an email to a recipient with an email address.","arguments":{"type":"object","properties":{"address":{"type":"string","description":"The recipient's email address.","pattern":"[a-z][a-z0-9]{1,10}@gmail\\\\.com","examples":["blob@gmail.com"]},"subject":{"type":"string"},"body":{"type":"string"}},"required":["address","subject","body"]}}]
      """
    )
  }

  @Test
  func `Uses Canonical Tool And Schema Field Order`() {
    let prompt = NeedlePrompt(
      system: "",
      user: "Weather?",
      tools: [.getWeather]
    )

    let rendered = prompt.formatted()
    let jsonStart = rendered.firstIndex(of: "[") ?? rendered.endIndex
    let jsonSlice = rendered[jsonStart...]

    let expected =
      #"[{"name":"get_weather","description":"Gets the current weather for a location.","arguments":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"],"additionalProperties":false}}]"#
    expectNoDifference(String(jsonSlice), expected)
  }
}
