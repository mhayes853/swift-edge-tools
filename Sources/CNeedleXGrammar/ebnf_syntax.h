#pragma once
#include <string>
#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <sstream>

constexpr std::string EBNF_LINE_SEPARATOR = " ::= ";

inline void replace_all(std::string& text, const std::string& needle, const std::string& replacement) {
    if (needle.empty()) return;
    size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        text.replace(pos, needle.size(), replacement);
        pos += replacement.size();
    }
}

struct EbnfSyntax {
    using RuleExpression = std::string;

    std::unordered_map<std::string, RuleExpression> rules;

    static std::string escape_string_literal(const std::string& text) {
        std::string escaped;
        escaped.reserve(text.size());
        for (char c : text) {
            if (c == '\\' || c == '"') escaped.push_back('\\');
            escaped.push_back(c);
        }
        return escaped;
    }

    static std::string escape_multiline_literal(const std::string& text) {
        std::string escaped = EbnfSyntax::escape_string_literal(text);
        replace_all(escaped, "\n", "\\n");
        return escaped;
    };

    static EbnfSyntax from_string(const std::string& ebnf) {
        EbnfSyntax parsed;

        std::istringstream stream(ebnf);
        std::string line;
        while (std::getline(stream, line)) {
            if (line.empty()) continue;

            const size_t separator_index = line.find(EBNF_LINE_SEPARATOR);
            if (separator_index == std::string::npos) {
                throw std::runtime_error("Invalid EBNF rule line: " + line);
            }
            const auto rule_name = line.substr(0, separator_index);
            const auto rule_expression = line.substr(separator_index + EBNF_LINE_SEPARATOR.size());
            parsed.rules[rule_name] = rule_expression;
        }
        return parsed;
    }

    void merge_with(const std::vector<std::pair<std::string, EbnfSyntax>>& others) {
        std::unordered_map<std::string, RuleExpression> merged_expr_by_name;
        UniqueNameGenerator name_generator;

        for (const auto& [identifier, syntax] : others) {
            EbnfSyntax incoming = syntax;
            std::unordered_map<std::string, std::string> rename_map;

            for (const auto& [rule_name, rule_expression] : incoming.rules) {
                auto existing = merged_expr_by_name.find(rule_name);
                if (existing != merged_expr_by_name.end() && existing->second != rule_expression) {
                    rename_map.emplace(rule_name, name_generator.next(identifier, rule_name));
                }
            }

            incoming.rename_rules(rename_map);

            for (const auto& [rule_name, rule_expression] : incoming.rules) {
                auto existing = merged_expr_by_name.find(rule_name);
                if (existing != merged_expr_by_name.end()) continue;
                merged_expr_by_name.emplace(rule_name, rule_expression);
                name_generator.names.insert(rule_name);
                rules[rule_name] = rule_expression;
            }
        }
    }

    void rename_rules(const std::unordered_map<std::string, std::string>& rename_map) {
        if (rename_map.empty()) return;

        std::unordered_map<std::string, RuleExpression> renamed_rules;
        renamed_rules.reserve(rules.size());

        for (const auto& [rule_name, rule_expression] : rules) {
            const auto renamed = rename_map.find(rule_name);
            const std::string& new_name = renamed != rename_map.end() ? renamed->second : rule_name;
            renamed_rules[new_name] = rewrite_rule_references(rule_expression, rename_map);
        }

        rules = std::move(renamed_rules);
    }

    std::string ebnf() {
        std::string out;
        for (const auto& [name, expresion] : rules) {
            out += name + EBNF_LINE_SEPARATOR + expresion + "\n";
        }
        return out;
    }

    void remove_json_whitespaces() {
        for (auto& [_, rule_expression] : rules) {
            replace_all(rule_expression, "\"\\n\"", "\"\"");
            replace_all(rule_expression, "\",\\n\"", "\",\"");
            replace_all(rule_expression, "\", \"", "\",\"");
            replace_all(rule_expression, "\": \"", "\":\"");
        }
    }

    void remove_unreachable_rules(const std::string& start_rule = "root") {
        if (!rules.contains(start_rule)) return;

        std::unordered_set<std::string> reachable;
        collect_reachable_rules(start_rule, reachable);

        std::erase_if(rules, [&](const auto& entry) {
            return !reachable.contains(entry.first);
        });
    }

private:
    static const std::regex& identifier_token_pattern() {
        static const std::regex token_pattern(
            R"((("(?:\\.|[^"\\])*")|(\[(?:\\.|[^\]\\])*\])|([A-Za-z_][A-Za-z0-9_]*)))"
        );
        return token_pattern;
    }

    static std::vector<std::string> referenced_rule_names(const std::string& expr) {
        std::vector<std::string> identifiers;
        std::sregex_iterator it(expr.begin(), expr.end(), identifier_token_pattern());
        const std::sregex_iterator end;
        for (; it != end; ++it) {
            if ((*it)[4].matched) {
                identifiers.push_back((*it)[4].str());
            }
        }
        return identifiers;
    }

    void collect_reachable_rules(const std::string& rule_name, std::unordered_set<std::string>& reachable) const {
        if (reachable.contains(rule_name)) return;

        reachable.insert(rule_name);
        const auto rule_it = rules.find(rule_name);
        if (rule_it == rules.end()) return;

        for (const auto& identifier : referenced_rule_names(rule_it->second)) {
            if (rules.contains(identifier)) {
                collect_reachable_rules(identifier, reachable);
            }
        }
    }

    struct UniqueNameGenerator {
        std::unordered_set<std::string> names;

        std::string next(const std::string& identifier, const std::string& name) {
            const auto combined_name = identifier + "__" + name;
            if (!names.contains(combined_name)) return combined_name;

            size_t suffix = 1;
            while (true) {
                const std::string candiate_name = combined_name + "_" + std::to_string(suffix);
                if (!names.contains(candiate_name)) return candiate_name;
                suffix++;
            }
        }
    };

    std::string rewrite_rule_references(
        const std::string& expr,
        const std::unordered_map<std::string, std::string>& rename_map
    ) {
        std::string out;
        out.reserve(expr.size());

        std::sregex_iterator it(expr.begin(), expr.end(), identifier_token_pattern());
        const std::sregex_iterator end;
        size_t last_pos = 0;
        for (; it != end; ++it) {
            const auto& match = *it;
            out += expr.substr(last_pos, static_cast<size_t>(match.position()) - last_pos);

            if (match[4].matched) {
                const std::string identifier = match[4].str();
                auto rename = rename_map.find(identifier);
                out += rename != rename_map.end() ? rename->second : identifier;
            } else {
                out += match.str();
            }

            last_pos = static_cast<size_t>(match.position() + match.length());
        }

        out += expr.substr(last_pos);
        return out;
    }
};
