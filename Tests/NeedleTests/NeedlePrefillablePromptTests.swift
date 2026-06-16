import CustomDump
import Needle
import Testing

@Suite
struct `NeedlePrefillablePrompt tests` {
  @Test
  func `Formats Properly`() {
    let prompt = NeedlePrefillablePrompt(
      system: "You are a helpful assistant who can send emails.",
      user: "Send an email to Henry."
    )

    expectNoDifference(
      prompt.formatted(),
      """
      You are a helpful assistant who can send emails.

      Send an email to Henry.
      """
    )
  }

  @Test
  func `Omits Separator When System Is Empty`() {
    let prompt = NeedlePrefillablePrompt(system: "", user: "Send an email to Henry.")

    expectNoDifference(
      prompt.formatted(),
      "Send an email to Henry."
    )
  }

  @Test
  func `Omits Separator When User Is Empty`() {
    let prompt = NeedlePrefillablePrompt(
      system: "You are a helpful assistant who can send emails.",
      user: ""
    )

    expectNoDifference(
      prompt.formatted(),
      "You are a helpful assistant who can send emails."
    )
  }
}
