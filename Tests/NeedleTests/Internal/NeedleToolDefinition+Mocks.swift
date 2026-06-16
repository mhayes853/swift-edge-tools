import Needle

extension NeedleToolDefinition {
  static let sendEmail = Self(
    name: "sendEmail",
    description: "Sends an email to someone.",
    arguments: .object(
      properties: [
        "address": .string(description: "The recipient's email address."),
        "subject": .string(),
        "body": .string()
      ],
      required: ["address", "subject", "body"]
    )
  )

  static let getWeather = Self(
    name: "getWeather",
    description: "Gets the current weather for a location.",
    arguments: .object(
      properties: ["location": .string()],
      required: ["location"],
      additionalProperties: .boolean(false)
    )
  )

  static let complexTool = Self(
    name: "complexTool",
    description: "A tool with broad parameter coverage.",
    arguments: .object(
      properties: [
        "title": .string(),
        "count": .number(),
        "enabled": .bool(),
        "mode": .string(enum: [.string("dry_run"), .string("execute")]),
        "ticket_id": .string(pattern: "[A-Z]{3}-[0-9]{2}"),
        "priority": .union(string: .string(), integer: .integer()),
        "routing": .object(
          properties: [
            "region": .string()
          ],
          required: ["region"],
          additionalProperties: .boolean(false)
        ),
        "labels": .object(
          additionalProperties: .boolean(false),
          patternProperties: [
            "[A-Z_]+": .integer()
          ]
        ),
        "window": .integer(minimum: 1, maximum: 5),
        "tuple_args": .array(
          items: .boolean(false),
          prefixItems: [.string(), .integer(), .bool()]
        ),
        "optional_note": .union(string: .string(), null: true),
        "tags": .array(
          items: .string(),
          uniqueItems: true
        ),
        "config": .object(
          properties: [
            "threshold": .number(),
            "flags": .array(items: .bool())
          ],
          required: ["threshold", "flags"],
          additionalProperties: .boolean(false)
        )
      ],
      required: [
        "title", "count", "enabled", "mode", "ticket_id", "priority",
        "routing", "labels", "window", "tuple_args", "optional_note",
        "tags", "config"
      ],
      additionalProperties: .boolean(false)
    )
  )
}
