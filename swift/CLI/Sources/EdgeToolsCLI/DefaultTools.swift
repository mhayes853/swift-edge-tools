import EdgeTools

public let defaultToolDefinitions: [EdgeToolDefinition] = [
  EdgeToolDefinition(
    name: "send_email",
    description: "Send an email to a recipient.",
    arguments: EdgeToolsGenerationSchema(
      .type(.object),
      .properties([
        "to": EdgeToolsGenerationSchema(.string, .description("The recipient's email address.")),
        "subject": EdgeToolsGenerationSchema(.string, .description("The subject line.")),
        "body": EdgeToolsGenerationSchema(.string, .description("The body of the email."))
      ]),
      .required(["to", "subject", "body"])
    )
  ),
  EdgeToolDefinition(
    name: "set_timer",
    description: "Set a timer for a duration or a target time.",
    arguments: EdgeToolsGenerationSchema(
      .type(.object),
      .properties([
        "duration": EdgeToolsGenerationSchema(
          .string,
          .description("A duration or end time, such as '20 minutes' or 'at 13:30'.")
        )
      ]),
      .required(["duration"])
    )
  ),
  EdgeToolDefinition(
    name: "search_web",
    description: "Search the web and return matching results.",
    arguments: EdgeToolsGenerationSchema(
      .type(.object),
      .properties([
        "query": EdgeToolsGenerationSchema(.string, .description("What to search for.")),
        "max_results": EdgeToolsGenerationSchema(
          .integer,
          .description("How many results to return.")
        )
      ]),
      .required(["query"])
    )
  ),
  EdgeToolDefinition(
    name: "create_calendar_event",
    description: "Create a calendar event.",
    arguments: EdgeToolsGenerationSchema(
      .type(.object),
      .properties([
        "title": EdgeToolsGenerationSchema(.string, .description("The title of the event.")),
        "start": EdgeToolsGenerationSchema(
          .string,
          .description("When the event starts, such as 'tomorrow at 9am'.")
        ),
        "attendees": EdgeToolsGenerationSchema(
          .type(.array),
          .items(.string),
          .description("Email addresses to invite.")
        ),
        "all_day": EdgeToolsGenerationSchema(
          .boolean,
          .description("Whether the event lasts the whole day.")
        )
      ]),
      .required(["title", "start"])
    )
  ),
  EdgeToolDefinition(
    name: "get_weather",
    description: "Get the current weather for a location.",
    arguments: EdgeToolsGenerationSchema(
      .type(.object),
      .properties([
        "location": EdgeToolsGenerationSchema(
          .string,
          .description("The city or place to look up.")
        ),
        "unit": EdgeToolsGenerationSchema(
          .string,
          .enum(["celsius", "fahrenheit"]),
          .description("The temperature unit to report.")
        )
      ]),
      .required(["location"])
    )
  )
]
