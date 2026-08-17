#include "bridging.h"

#include <algorithm>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <minja/minja.hpp>

namespace {

constexpr const char *kNowContextKey = "edge_tools_now";

thread_local std::string last_error;

void set_last_error(const std::string &message) { last_error = message; }

// Rewrites `{% generation %}` / `{% endgeneration %}` into always-true conditionals while
// preserving whitespace-control markers. Assistant-token masking has no meaning when
// rendering a prompt.
std::string generation_blocks_neutralized(const std::string &source) {
  std::string rewritten;
  rewritten.reserve(source.size());
  size_t cursor = 0;
  while (true) {
    size_t start = source.find("{%", cursor);
    if (start == std::string::npos) {
      break;
    }
    size_t inner_end = source.find("%}", start + 2);
    if (inner_end == std::string::npos) {
      break;
    }
    size_t end = inner_end + 2;
    std::string inner = source.substr(start + 2, inner_end - (start + 2));

    auto is_marker = [](char character) { return character == '-' || character == '+'; };
    std::string opening;
    std::string closing;
    if (!inner.empty() && is_marker(inner.front())) {
      opening = inner.front();
    }
    if (!inner.empty() && is_marker(inner.back())) {
      closing = inner.back();
    }
    size_t body_begin = 0;
    size_t body_end = inner.size();
    while (body_begin < body_end &&
           (std::isspace(static_cast<unsigned char>(inner[body_begin])) ||
            is_marker(inner[body_begin]))) {
      ++body_begin;
    }
    while (body_end > body_begin &&
           (std::isspace(static_cast<unsigned char>(inner[body_end - 1])) ||
            is_marker(inner[body_end - 1]))) {
      --body_end;
    }
    std::string body = inner.substr(body_begin, body_end - body_begin);

    const char *replacement = nullptr;
    if (body == "generation") {
      replacement = "if true";
    } else if (body == "endgeneration") {
      replacement = "endif";
    }

    rewritten.append(source, cursor, start - cursor);
    if (replacement != nullptr) {
      rewritten += "{%";
      rewritten += opening;
      rewritten += ' ';
      rewritten += replacement;
      rewritten += ' ';
      rewritten += closing;
      rewritten += "%}";
    } else {
      rewritten.append(source, start, end - start);
    }
    cursor = end;
  }
  rewritten.append(source, cursor, source.size() - cursor);
  return rewritten;
}

std::time_t rendering_instant(nlohmann::ordered_json &context) {
  if (!context.contains(kNowContextKey)) {
    return std::time(nullptr);
  }
  auto pinned = context[kNowContextKey];
  context.erase(kNowContextKey);
  if (!pinned.is_number_integer()) {
    throw std::runtime_error(
        "`edge_tools_now` must be whole seconds since the Unix epoch.");
  }
  return static_cast<std::time_t>(pinned.get<int64_t>());
}

// Serializes a value the way Python's json.dumps does: ", " and ": " separators in
// compact mode, insertion-ordered keys, and non-ASCII passed through unless
// `ensure_ascii` asks for escapes. Scalars reuse nlohmann's writer, which matches
// json.dumps escaping for strings and shortest round-trip formatting for numbers.
void write_python_json(
    const nlohmann::ordered_json &value,
    std::string &out,
    bool ensure_ascii,
    int64_t indent,
    bool sort_keys,
    size_t level) {
  auto write_break = [&](size_t break_level) {
    if (indent >= 0) {
      out += '\n';
      out.append(break_level * static_cast<size_t>(indent), ' ');
    }
  };
  auto write_item_separator = [&](size_t item_level) {
    out += ',';
    if (indent >= 0) {
      write_break(item_level);
    } else {
      out += ' ';
    }
  };

  if (value.is_object()) {
    if (value.empty()) {
      out += "{}";
      return;
    }
    std::vector<std::string> keys;
    keys.reserve(value.size());
    for (auto it = value.begin(); it != value.end(); ++it) {
      keys.push_back(it.key());
    }
    if (sort_keys) {
      std::sort(keys.begin(), keys.end());
    }
    out += '{';
    write_break(level + 1);
    for (size_t index = 0; index < keys.size(); ++index) {
      if (index > 0) {
        write_item_separator(level + 1);
      }
      out += nlohmann::ordered_json(keys[index]).dump(-1, ' ', ensure_ascii);
      out += ": ";
      write_python_json(value.at(keys[index]), out, ensure_ascii, indent, sort_keys, level + 1);
    }
    write_break(level);
    out += '}';
  } else if (value.is_array()) {
    if (value.empty()) {
      out += "[]";
      return;
    }
    out += '[';
    write_break(level + 1);
    for (size_t index = 0; index < value.size(); ++index) {
      if (index > 0) {
        write_item_separator(level + 1);
      }
      write_python_json(value.at(index), out, ensure_ascii, indent, sort_keys, level + 1);
    }
    write_break(level);
    out += ']';
  } else {
    out += value.dump(-1, ' ', ensure_ascii);
  }
}

minja::Value tojson_value(const std::shared_ptr<minja::Context> &, minja::ArgumentsValue &args) {
  if (args.args.empty()) {
    throw std::runtime_error("tojson expects a value to serialize");
  }
  int64_t indent = -1;
  bool ensure_ascii = false;
  bool sort_keys = false;
  if (args.args.size() > 1) {
    indent = args.args[1].get<int64_t>();
  }
  if (args.args.size() > 2) {
    throw std::runtime_error("tojson expects at most two positional arguments");
  }
  for (auto &kwarg : args.kwargs) {
    if (kwarg.first == "indent") {
      indent = kwarg.second.is_null() ? -1 : kwarg.second.get<int64_t>();
    } else if (kwarg.first == "ensure_ascii") {
      ensure_ascii = kwarg.second.get<bool>();
    } else if (kwarg.first == "sort_keys") {
      sort_keys = kwarg.second.get<bool>();
    } else {
      throw std::runtime_error("Unknown argument " + kwarg.first + " for function tojson");
    }
  }
  auto serialized = nlohmann::ordered_json::parse(args.args[0].dump(-1, /* to_json= */ true));
  std::string out;
  write_python_json(serialized, out, ensure_ascii, indent, sort_keys, 0);
  return minja::Value(out);
}

// jinja2's `min` / `max` filters over a sequence, which minja does not provide. Both
// compare numerically when every element is a number and lexicographically otherwise,
// matching Python's ordering for the scalar sequences chat templates use.
minja::Value extremum_value(const char *name, bool wantsMinimum, minja::ArgumentsValue &args) {
  args.expectArgs(name, {1, 1}, {0, 0});
  auto &sequence = args.args[0];
  if (!sequence.is_array()) {
    throw std::runtime_error(std::string(name) + " expects a sequence");
  }
  if (sequence.size() == 0) {
    return minja::Value();
  }
  auto isLess = [](const minja::Value &lhs, const minja::Value &rhs) {
    if (lhs.is_number() && rhs.is_number()) {
      return lhs.get<double>() < rhs.get<double>();
    }
    return lhs.to_str() < rhs.to_str();
  };
  auto extremum = sequence.at(size_t{0});
  for (size_t index = 1; index < sequence.size(); ++index) {
    auto element = sequence.at(index);
    if (isLess(element, extremum) == wantsMinimum) {
      extremum = element;
    }
  }
  return extremum;
}

std::string formatted_utc(std::time_t instant, const std::string &format) {
  std::tm components{};
#ifdef _WIN32
  gmtime_s(&components, &instant);
#else
  gmtime_r(&instant, &components);
#endif
  std::ostringstream out;
  out << std::put_time(&components, format.c_str());
  return out.str();
}

std::string render(const std::string &raw_source, const std::string &context_json) {
  auto source = generation_blocks_neutralized(raw_source);
  auto context_values = nlohmann::ordered_json::parse(context_json);
  if (!context_values.is_object()) {
    throw std::runtime_error("The template context must be a JSON object.");
  }
  auto instant = rendering_instant(context_values);

  auto root = minja::Parser::parse(
      source,
      {/* .trim_blocks = */ true,
       /* .lstrip_blocks = */ true,
       /* .keep_trailing_newline = */ false});
  auto context = minja::Context::make(minja::Value(context_values));
  context->set(
      "strftime_now",
      minja::Value::callable([instant](const std::shared_ptr<minja::Context> &,
                                       minja::ArgumentsValue &args) {
        args.expectArgs("strftime_now", {1, 1}, {0, 0});
        return minja::Value(formatted_utc(instant, args.args[0].get<std::string>()));
      }));
  context->set("tojson", minja::Value::callable(tojson_value));
  context->set(
      "min",
      minja::Value::callable([](const std::shared_ptr<minja::Context> &,
                                minja::ArgumentsValue &args) {
        return extremum_value("min", /* wantsMinimum= */ true, args);
      }));
  context->set(
      "max",
      minja::Value::callable([](const std::shared_ptr<minja::Context> &,
                                minja::ArgumentsValue &args) {
        return extremum_value("max", /* wantsMinimum= */ false, args);
      }));
  return root->render(context);
}

int32_t fill(
    uint8_t *text, size_t text_capacity, size_t *text_count, const std::string &value) {
  *text_count = value.size();
  if (value.size() > text_capacity) {
    set_last_error("An output buffer was too small for the result.");
    return EDGE_TEMPLATE_BUFFER_TOO_SMALL;
  }
  if (!value.empty()) {
    std::memcpy(text, value.data(), value.size());
  }
  set_last_error("");
  return EDGE_TEMPLATE_SUCCESS;
}

}  // namespace

extern "C" {

const char *edge_template_last_error_message(void) { return last_error.c_str(); }

int32_t edge_template_render(
    const uint8_t *source,
    size_t source_count,
    const uint8_t *context_json,
    size_t context_json_count,
    uint8_t *text,
    size_t text_capacity,
    size_t *text_count) {
  if (source == nullptr) {
    set_last_error("A template source is required.");
    return EDGE_TEMPLATE_INVALID_ARGUMENT;
  }
  if (context_json == nullptr) {
    set_last_error("A template context is required.");
    return EDGE_TEMPLATE_INVALID_ARGUMENT;
  }
  if (text_count == nullptr || (text == nullptr && text_capacity > 0)) {
    set_last_error("An output buffer is required.");
    return EDGE_TEMPLATE_INVALID_ARGUMENT;
  }
  try {
    auto rendered = render(
        std::string(reinterpret_cast<const char *>(source), source_count),
        std::string(reinterpret_cast<const char *>(context_json), context_json_count));
    return fill(text, text_capacity, text_count, rendered);
  } catch (const std::exception &error) {
    set_last_error(error.what());
    return EDGE_TEMPLATE_FAILURE;
  } catch (...) {
    set_last_error("The template renderer encountered an unexpected failure.");
    return EDGE_TEMPLATE_FAILURE;
  }
}

}  // extern "C"
