import EdgeTools
import Foundation

// MARK: - Default Tools

/// A small general-purpose toolset used when `--tools` is not given, covering strings, integers,
/// booleans, enums and arrays so a model's argument handling gets exercised.
public let defaultToolDefinitions: [EdgeToolDefinition] = {
  guard let file = try? ToolsFile(data: Data(defaultToolsJSON.utf8)) else {
    preconditionFailure("The built-in toolset is not valid JSON.")
  }
  return file.definitions
}()

private let defaultToolsJSON = """
  [
    {
      "name": "send_email",
      "description": "Send an email to a recipient.",
      "parameters": {
        "type": "object",
        "properties": {
          "to": { "type": "string", "description": "The recipient's email address." },
          "subject": { "type": "string", "description": "The subject line." },
          "body": { "type": "string", "description": "The body of the email." }
        },
        "required": ["to", "subject", "body"]
      }
    },
    {
      "name": "set_timer",
      "description": "Set a timer for a duration or a target time.",
      "parameters": {
        "type": "object",
        "properties": {
          "duration": {
            "type": "string",
            "description": "A duration or end time, such as '20 minutes' or 'at 13:30'."
          }
        },
        "required": ["duration"]
      }
    },
    {
      "name": "search_web",
      "description": "Search the web and return matching results.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": { "type": "string", "description": "What to search for." },
          "max_results": {
            "type": "integer",
            "description": "How many results to return."
          }
        },
        "required": ["query"]
      }
    },
    {
      "name": "create_calendar_event",
      "description": "Create a calendar event.",
      "parameters": {
        "type": "object",
        "properties": {
          "title": { "type": "string", "description": "The title of the event." },
          "start": {
            "type": "string",
            "description": "When the event starts, such as 'tomorrow at 9am'."
          },
          "attendees": {
            "type": "array",
            "items": { "type": "string" },
            "description": "Email addresses to invite."
          },
          "all_day": {
            "type": "boolean",
            "description": "Whether the event lasts the whole day."
          }
        },
        "required": ["title", "start"]
      }
    },
    {
      "name": "get_weather",
      "description": "Get the current weather for a location.",
      "parameters": {
        "type": "object",
        "properties": {
          "location": { "type": "string", "description": "The city or place to look up." },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "The temperature unit to report."
          }
        },
        "required": ["location"]
      }
    }
  ]
  """
