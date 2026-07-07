import Needle

extension NeedleToolDefinition {
  static let sendEmail = Self(
    name: "sendEmail",
    description: "Sends an email to a recipient with an email address.",
    arguments: NeedleGenerationSchema(
      .type(.object),
      .properties([
        "address": NeedleGenerationSchema(
          .string,
          .description("The recipient's email address."),
          .pattern("[a-z][a-z0-9]{1,10}@gmail\\.com"),
          .examples(["blob@gmail.com"])
        ),
        "subject": .string,
        "body": .string,
      ]),
      .required(["address", "subject", "body"])
    )
  )

  static let getWeather = Self(
    name: "getWeather",
    description: "Gets the current weather for a location.",
    arguments: NeedleGenerationSchema(
      .type(.object),
      .properties(["location": .string]),
      .required(["location"]),
      .additionalProperties(false)
    )
  )

  static let complexTool = Self(
    name: "complexTool",
    description: "A tool with broad parameter coverage.",
    arguments: NeedleGenerationSchema(
      .type(.object),
      .properties([
        "title": .string,
        "count": .number,
        "enabled": .boolean,
        "mode": NeedleGenerationSchema(
          .string,
          .enum([.string("dry_run"), .string("execute")])
        ),
        "ticket_id": NeedleGenerationSchema(
          .string,
          .pattern("[A-Z]{3}-[0-9]{2}")
        ),
        "priority": NeedleGenerationSchema(
          .type([.string, .integer])
        ),
        "routing": NeedleGenerationSchema(
          .type(.object),
          .properties([
            "region": .string
          ]),
          .required(["region"]),
          .additionalProperties(false)
        ),
        "labels": NeedleGenerationSchema(
          .type(.object),
          .additionalProperties(false),
          .patternProperties([
            "[A-Z_]+": .integer
          ])
        ),
        "window": NeedleGenerationSchema(
          .integer,
          .minimum(1),
          .maximum(5)
        ),
        "tuple_args": NeedleGenerationSchema(
          .type(.array),
          .items(false),
          .prefixItems([.string, .integer, .boolean])
        ),
        "optional_note": NeedleGenerationSchema(
          .type([.string, .null])
        ),
        "tags": NeedleGenerationSchema(
          .type(.array),
          .items(.string),
          .uniqueItems()
        ),
        "config": NeedleGenerationSchema(
          .type(.object),
          .properties([
            "threshold": .number,
            "flags": NeedleGenerationSchema(
              .type(.array),
              .items(.boolean)
            )
          ]),
          .required(["threshold", "flags"]),
          .additionalProperties(false)
        )
      ]),
      .required([
        "title", "count", "enabled", "mode", "ticket_id", "priority",
        "routing", "labels", "window", "tuple_args", "optional_note",
        "tags", "config"
      ]),
      .additionalProperties(false)
    )
  )
}
