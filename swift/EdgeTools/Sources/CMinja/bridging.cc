#include "bridging.h"

#include <algorithm>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <minja/minja.hpp>

namespace {

constexpr const char *kNowContextKey = "edge_tools_now";

thread_local std::string last_error;

void set_last_error(const std::string &message) { last_error = message; }

size_t fail(const std::string &message) {
  set_last_error(message);
  return 0;
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

void write_python_json(
    const nlohmann::ordered_json &value,
    std::string &out,
    bool ensure_ascii,
    int64_t indent,
    bool sort_keys,
    size_t level) {
  if (!value.is_structured()) {
    out += value.dump(-1, ' ', ensure_ascii);
    return;
  }
  const char *brackets = value.is_object() ? "{}" : "[]";
  if (value.empty()) {
    out.append(brackets, 2);
    return;
  }

  auto write_break = [&](size_t break_level) {
    if (indent >= 0) {
      out += '\n';
      out.append(break_level * static_cast<size_t>(indent), ' ');
    }
  };
  bool first = true;
  auto write_separator = [&] {
    if (std::exchange(first, false)) {
      return;
    }
    out += ',';
    if (indent < 0) {
      out += ' ';
    } else {
      write_break(level + 1);
    }
  };

  out += brackets[0];
  write_break(level + 1);
  if (value.is_array()) {
    for (const auto &element : value) {
      write_separator();
      write_python_json(element, out, ensure_ascii, indent, sort_keys, level + 1);
    }
  } else {
    std::vector<std::string> keys;
    keys.reserve(value.size());
    for (auto member = value.begin(); member != value.end(); ++member) {
      keys.push_back(member.key());
    }
    if (sort_keys) {
      std::sort(keys.begin(), keys.end());
    }
    for (const auto &key : keys) {
      write_separator();
      out += nlohmann::ordered_json(key).dump(-1, ' ', ensure_ascii);
      out += ": ";
      write_python_json(value.at(key), out, ensure_ascii, indent, sort_keys, level + 1);
    }
  }
  write_break(level);
  out += brackets[1];
}

minja::Value tojson_value(const std::shared_ptr<minja::Context> &, minja::ArgumentsValue &args) {
  if (args.args.empty()) {
    throw std::runtime_error("tojson expects a value to serialize");
  }
  if (args.args.size() > 2) {
    throw std::runtime_error("tojson expects at most two positional arguments");
  }
  int64_t indent = args.args.size() > 1 ? args.args[1].get<int64_t>() : -1;
  bool ensure_ascii = false;
  bool sort_keys = false;
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
  std::string out;
  write_python_json(
      args.args[0].get<nlohmann::ordered_json>(), out, ensure_ascii, indent, sort_keys, 0);
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

std::shared_ptr<minja::TemplateNode> parsed_template(const std::string &source) {
  constexpr size_t max_cached_templates = 4;
  static std::mutex mutex;
  static std::vector<std::pair<std::string, std::shared_ptr<minja::TemplateNode>>> cache;

  std::lock_guard<std::mutex> guard(mutex);
  auto cached = std::find_if(cache.begin(), cache.end(), [&source](const auto &entry) {
    return entry.first == source;
  });
  if (cached != cache.end()) {
    std::rotate(cache.begin(), cached, cached + 1);
    return cache.front().second;
  }
  auto root = minja::Parser::parse(source, { true, true, false});
  cache.insert(cache.begin(), { source, root });
  if (cache.size() > max_cached_templates) {
    cache.pop_back();
  }
  return root;
}

std::string render(const std::string &source, const std::string &context_json) {
  auto context_values = nlohmann::ordered_json::parse(context_json);
  if (!context_values.is_object()) {
    throw std::runtime_error("The template context must be a JSON object.");
  }
  auto instant = rendering_instant(context_values);

  auto root = parsed_template(source);
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

size_t write_string(const std::string &value, char *text, size_t text_capacity) {
  const size_t required_capacity = value.size() + 1;
  if (text != nullptr && text_capacity >= required_capacity) {
    std::memcpy(text, value.c_str(), required_capacity);
  }
  return required_capacity;
}

}  // namespace

extern "C" {

const char *edge_template_last_error_message(void) { return last_error.c_str(); }

size_t edge_template_render(
    const char *source, const char *context_json, char *text, size_t text_capacity) {
  if (source == nullptr) {
    return fail("A template source is required.");
  }
  if (context_json == nullptr) {
    return fail("A template context is required.");
  }
  try {
    auto rendered = render(source, context_json);
    set_last_error("");
    return write_string(rendered, text, text_capacity);
  } catch (const std::exception &error) {
    return fail(error.what());
  } catch (...) {
    return fail("The template renderer encountered an unexpected failure.");
  }
}

}  // extern "C"
