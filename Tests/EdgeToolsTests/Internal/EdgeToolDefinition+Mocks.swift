import EdgeTools

extension EdgeToolDefinition {
  static let sendEmail = Self(
    name: "sendEmail",
    description: "Sends an email to a recipient with an email address.",
    arguments: EdgeToolsGenerationSchema(
      .type(.object),
      .properties([
        "address": EdgeToolsGenerationSchema(
          .string,
          .description("The recipient's email address."),
          .pattern("[a-z][a-z0-9]{1,10}@gmail\\.com"),
          .examples(["blob@gmail.com"])
        ),
        "subject": .string,
        "body": .string
      ]),
      .required(["address", "subject", "body"])
    ),
    includesSchemaInInstructions: true
  )

  static let getWeather = Self(
    name: "getWeather",
    description: "Gets the current weather for a location.",
    arguments: EdgeToolsGenerationSchema(
      .type(.object),
      .properties(["location": .string]),
      .required(["location"]),
      .additionalProperties(false)
    ),
    includesSchemaInInstructions: true
  )

  static let integerTool = Self(
    name: "integerTool",
    description: "Accepts an integer.",
    arguments: EdgeToolsGenerationSchema(
      .type(.object),
      .properties(["value": .integer]),
      .required(["value"]),
      .additionalProperties(false)
    )
  )

  static let complexTool = Self(
    name: "complexTool",
    description: "A tool with broad parameter coverage.",
    arguments: EdgeToolsGenerationSchema(
      .type(.object),
      .properties([
        "title": .string,
        "count": .number,
        "enabled": .boolean,
        "mode": EdgeToolsGenerationSchema(
          .string,
          .enum([.string("dry_run"), .string("execute")])
        ),
        "ticket_id": EdgeToolsGenerationSchema(
          .string,
          .pattern("[A-Z]{3}-[0-9]{2}")
        ),
        "priority": EdgeToolsGenerationSchema(
          .type([.string, .integer])
        ),
        "routing": EdgeToolsGenerationSchema(
          .type(.object),
          .properties([
            "region": .string
          ]),
          .required(["region"]),
          .additionalProperties(false)
        ),
        "labels": EdgeToolsGenerationSchema(
          .type(.object),
          .additionalProperties(false),
          .patternProperties([
            "[A-Z_]+": .integer
          ])
        ),
        "window": EdgeToolsGenerationSchema(
          .integer,
          .minimum(1),
          .maximum(5)
        ),
        "tuple_args": EdgeToolsGenerationSchema(
          .type(.array),
          .items(false),
          .prefixItems([.string, .integer, .boolean])
        ),
        "optional_note": EdgeToolsGenerationSchema(
          .type([.string, .null])
        ),
        "tags": EdgeToolsGenerationSchema(
          .type(.array),
          .items(.string),
          .uniqueItems()
        ),
        "config": EdgeToolsGenerationSchema(
          .type(.object),
          .properties([
            "threshold": .number,
            "flags": EdgeToolsGenerationSchema(
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
    ),
    includesSchemaInInstructions: true
  )
}
