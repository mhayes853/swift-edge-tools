import CustomDump
import EdgeToolsCLI
import Foundation
import Testing

@Suite
struct `GrammarOption tests` {
  @Test
  func `Parses Named Constraints`() {
    expectNoDifference(GrammarOption(argument: "auto"), .auto)
    expectNoDifference(GrammarOption(argument: "unconstrained"), .unconstrained)
  }

  @Test
  func `Infers The Format From A File Extension`() {
    expectNoDifference(
      GrammarOption(argument: "/tmp/calls.ebnf"),
      .custom(format: .ebnf, source: .file(URL(fileURLWithPath: "/tmp/calls.ebnf")))
    )
    expectNoDifference(
      GrammarOption(argument: "/tmp/calls.lark"),
      .custom(format: .lark, source: .file(URL(fileURLWithPath: "/tmp/calls.lark")))
    )
    expectNoDifference(
      GrammarOption(argument: "/tmp/response.json"),
      .custom(format: .jsonSchema, source: .file(URL(fileURLWithPath: "/tmp/response.json")))
    )
  }

  @Test
  func `Parses Inline Grammars By Format Prefix`() {
    expectNoDifference(
      GrammarOption(argument: #"regex:\d{4}"#),
      .custom(format: .regex, source: .inline(#"\d{4}"#))
    )
    expectNoDifference(
      GrammarOption(argument: "structural-tag:{}"),
      .custom(format: .structuralTag, source: .inline("{}"))
    )
  }

  @Test
  func `Rejects Files Whose Format Cannot Be Inferred`() {
    expectNoDifference(GrammarOption(argument: "/tmp/calls.txt"), nil)
  }

  @Test
  func `Auto Constrains To Tool Calls`() throws {
    let constraint = try GrammarOption.auto.constraint(toolCallRange: .exact(1))
    expectNoDifference(constraint.toolCallRange, .exact(1))
  }

  @Test
  func `Custom Grammars Replace The Tool Call Grammar`() throws {
    let grammar = GrammarOption.custom(format: .regex, source: .inline("a+"))
    let constraint = try grammar.constraint(toolCallRange: .exact(1))

    expectNoDifference(constraint.toolCallRange, nil)
  }

  @Test
  func `Unconstrained Has No Tool Call Range`() throws {
    let constraint = try GrammarOption.unconstrained.constraint(toolCallRange: .exact(1))
    expectNoDifference(constraint.toolCallRange, nil)
  }
}
