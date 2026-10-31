import CustomDump
import EdgeToolsTemplateRendering
import Testing

@Suite
struct `EdgeToolsTemplateSyntax tests` {
  @Test
  func `Calls String Methods`() throws {
    expectNoDifference(try render("{{ ' a '.strip() }}"), "a")
    expectNoDifference(try render("{{ ' a '.lstrip() }}"), "a ")
    expectNoDifference(try render("{{ ' a '.rstrip() }}"), " a")
    expectNoDifference(try render("{{ 'abcXYZabc'.strip('ac') }}"), "bcXYZab")
    expectNoDifference(try render("{{ 'a b'.split(' ') | tojson }}"), #"["a", "b"]"#)
    expectNoDifference(try render("{{ 'ok'.capitalize() }}"), "Ok")
    expectNoDifference(
      try render("{{ 'abcXYZabcXYZabc'.replace('bc', 'oui') }}"),
      "aouiXYZaouiXYZaoui"
    )
    expectNoDifference(
      try render("{{ 'abcXYZabcXYZabc'.replace('abc', 'ok', 2) }}"),
      "okXYZokXYZabc"
    )
    expectNoDifference(
      try render("{{ 'abcXYZabcXYZabc'.replace('def', 'ok') }}"),
      "abcXYZabcXYZabc"
    )
    expectNoDifference(try render("{{ 'hello world'.upper() }}"), "HELLO WORLD")
    expectNoDifference(try render("{{ 'MiXeD'.upper() }}"), "MIXED")
    expectNoDifference(try render("{{ ''.upper() }}"), "")
    expectNoDifference(try render("{{ 'HELLO WORLD'.lower() }}"), "hello world")
    expectNoDifference(try render("{{ 'MiXeD'.lower() }}"), "mixed")
    expectNoDifference(try render("{{ ''.lower() }}"), "")
    expectNoDifference(try render("{{ 'foo bar'.title() }}"), "Foo Bar")
    expectNoDifference(
      try render("{{ 'abc'.startswith('ab') }},{{ ''.startswith('a') }}"),
      "True,False"
    )
    expectNoDifference(
      try render("{{ 'abc'.endswith('bc') }},{{ ''.endswith('a') }}"),
      "True,False"
    )
  }

  @Test
  func `Ignores Comments`() throws {
    expectNoDifference(
      try render("{# Hey\nHo #}{#- Multiline...\nComments! -#}{{ 'ok' }}{# yo #}"),
      "ok"
    )
  }

  @Test
  func `Applies Block Whitespace Options`() throws {
    expectNoDifference(
      try render("  {% set _ = 1 %}    {% set _ = 2 %}b", options: lstripTrimBlocks),
      "    b"
    )
    expectNoDifference(
      try render(
        "{%- if True %}        {% set _ = x %}{%- endif %}{{ 1 }}",
        options: lstripTrimBlocks
      ),
      "        1"
    )

    let ifBlock = "    {% if True %}\n    {% endif %}"
    expectNoDifference(try render(ifBlock, options: lstripBlocks), "\n")
    expectNoDifference(try render(ifBlock, options: lstripTrimBlocks), "")
    expectNoDifference(try render(ifBlock, options: trimBlocks), "        ")

    let paddedSet = "  {% set _ = 1 %}    "
    expectNoDifference(try render(paddedSet), "      ")
    expectNoDifference(try render(paddedSet, options: lstripBlocks), "    ")
    expectNoDifference(try render(paddedSet, options: trimBlocks), "      ")
    expectNoDifference(try render(paddedSet, options: lstripTrimBlocks), "    ")

    let multilineSet = "  \n    {% set _ = 1 %}        \n                "
    expectNoDifference(try render(multilineSet), "  \n            \n                ")
    expectNoDifference(
      try render(multilineSet, options: lstripBlocks),
      "  \n        \n                "
    )
    expectNoDifference(
      try render(multilineSet, options: trimBlocks),
      "  \n            \n                "
    )
    expectNoDifference(
      try render(multilineSet, options: lstripTrimBlocks),
      "  \n        \n                "
    )

    let leadingSet = "{% set _ = 1 %}\n  "
    expectNoDifference(try render(leadingSet), "\n  ")
    expectNoDifference(try render(leadingSet, options: lstripBlocks), "\n  ")
    expectNoDifference(try render(leadingSet, options: trimBlocks), "  ")
    expectNoDifference(try render(leadingSet, options: lstripTrimBlocks), "  ")

    let trimmed = "\n  {% if true %}Hello{% endif %}  \n...\n\n"
    expectNoDifference(try render(trimmed, options: trimBlocks), "\n  Hello  \n...\n")
    expectNoDifference(try render(trimmed), "\n  Hello  \n...\n")
    expectNoDifference(try render(trimmed, options: lstripBlocks), "\nHello  \n...\n")
    expectNoDifference(try render(trimmed, options: lstripTrimBlocks), "\nHello  \n...\n")

    expectNoDifference(try render("  {{- ' a\n'}}", options: trimBlocks), " a\n")
  }

  @Test
  func `Strips Whitespace Around Control Markers`() throws {
    expectNoDifference(try render("  {{- 'a' -}}{{ '  ' }}{{- 'b' -}}  "), "a  b")
    expectNoDifference(
      try render(
        "    {%- if True %}{%- endif %}{{ '        ' }}{%- for x in [] %}foo{% endfor %}end"
      ),
      "        end"
    )
    expectNoDifference(try render("a\nb\n"), "a\nb")
    expectNoDifference(try render(" a {{  'b' -}} c "), " a bc ")
    expectNoDifference(try render(" a {{- 'b'  }} c "), " ab c ")
    expectNoDifference(try render("a\n{{- 'b'  }}\nc"), "ab\nc")
    expectNoDifference(try render("a\n{{  'b' -}}\nc"), "a\nbc")
    expectNoDifference(try render(#" {{ "a" -}} b {{- "c" }} "#), " abc ")
  }

  @Test
  func `Applies Filters`() throws {
    expectNoDifference(try render("{{ range(*[2,4]) | list }}"), "[2, 3]")
    expectNoDifference(
      try render(
        "{% for i in [true, false, 10, -10, 10.1, -10.1, None, 'a', '2', {}, [1]] %}"
          + "{{ i | int }}, {% endfor %}"
      ),
      "1, 0, 10, -10, 10, -10, 0, 0, 2, 0, 0, "
    )
    expectNoDifference(try render("{% filter trim %} abc {% endfilter %}"), "abc")
    expectNoDifference(try render("{{ 'AbC' | lower }}"), "abc")
    expectNoDifference(try render("{{ 'me' | upper }}"), "ME")
    expectNoDifference(
      try render("{{ foo | default('the default') }}{{ 1 | default('nope') }}"),
      "the default1"
    )
    expectNoDifference(
      try render("{{ '' | default('the default', true) }}{{ 1 | default('nope', true) }}"),
      "the default1"
    )
    expectNoDifference(
      try render(
        "{% set txt = 'a\\nb\\n' %}{{ txt | indent(2) }}|{{ txt | indent(2, first=true) }}"
      ),
      "a\n  b\n|  a\n  b\n"
    )
    expectNoDifference(try render(#"{{ "123456789" | length }}"#), "9")
    expectNoDifference(try render("{{ 1 | safe }}"), "1")
    expectNoDifference(
      try render(#"{{ none | selectattr("foo", "equalto", "bar") | list }}"#),
      "[]"
    )
    expectNoDifference(
      try render("{{ ['a', 'b', 'c', 'a'] | select('in', ['a']) | list }}"),
      "['a', 'a']"
    )
    expectNoDifference(
      try render(#"{{ [{"a": 1}, {"a": 2}, {}] | selectattr("a", "equalto", 1) | list }}"#),
      "[{'a': 1}]"
    )
    expectNoDifference(
      try render(#"{{ [{"a": 1}, {"a": 2}, {}] | rejectattr("a", "equalto", 1) | list }}"#),
      "[{'a': 2}, {}]"
    )
    expectNoDifference(
      try render(#"{{ [{"a": 1}, {"a": 2}] | map(attribute="a") | list }}"#),
      "[1, 2]"
    )
    expectNoDifference(try render(#"{{ ["", "a"] | map("length") | list }}"#), "[0, 1]")
    expectNoDifference(try render("{{ range(3) | last }}"), "2")
    expectNoDifference(try render("{{ 'a' + [] | length | string + 'b' }}"), "a0b")
    expectNoDifference(try render("{{ [1, 2, 3] | join(', ') + '...' }}"), "1, 2, 3...")
    expectNoDifference(
      try render("{{ 'Tools: ' + [1, 2, 3] | reject('equalto', 2) | join(', ') + '...' }}"),
      "Tools: 1, 3..."
    )
    expectNoDifference(
      try render("{{ 'Tools: ' + [1, 2, 3] | select('equalto', 2) | join(', ') + '...' }}"),
      "Tools: 2..."
    )
    expectNoDifference(try render("{{ [1, 2, 3] | join(', ') }}"), "1, 2, 3")
    expectNoDifference(
      try render("{{ [1, False, 2, '3', 1, '3', False] | unique | list }}"),
      "[1, False, 2, '3']"
    )
    expectNoDifference(try render("{{ range(5) | length % 2 }}"), "1")
    expectNoDifference(
      try render("{{ range(5) | length % 2 == 1 }},{{ [] | length > 0 }}"),
      "True,False"
    )
    expectNoDifference(try render("{{ {1: 2} | items | list | tojson }}"), "[[1, 2]]")
    expectNoDifference(
      try render("{{ {1: 2, 3: 4, 5: 7} | dictsort | tojson }}"),
      "[[1, 2], [3, 4], [5, 7]]"
    )
    expectNoDifference(try render(#"{{ {1: 2}.items() | map("list") | list }}"#), "[[1, 2]]")
    expectNoDifference(try render("{{ ' a  ' | trim }}"), "a")
    expectNoDifference(try render("{{ None | trim }}"), "")
    expectNoDifference(
      try render(
        lines(
          "",
          "            {%- set res = [] -%}",
          #"            {%- for c in ["<", ">", "&", '"'] -%}"#,
          "                {%- set _ = res.append(c | e) -%}",
          "            {%- endfor -%}",
          #"            {{- res | join(", ") -}}"#,
          "        "
        )
      ),
      "&lt;, &gt;, &amp;, &#34;"
    )
    expectNoDifference(
      try render(
        lines(
          "{%- set separator = joiner(' | ') -%}",
          #"            {%- for item in ["a", "b", "c"] %}{{ separator() }}{{ item }}{% endfor -%}"#
        )
      ),
      "a | b | c"
    )
  }

  @Test
  func `Evaluates Operators And Tests`() throws {
    expectNoDifference(try render("{{ [1] + [2, 3] }}"), "[1, 2, 3]")
    expectNoDifference(try render("{{ {} is mapping }},{{ '' is mapping }}"), "True,False")
    expectNoDifference(try render("{{ {} is iterable }},{{ '' is iterable }}"), "True,True")
    expectNoDifference(try render(#"{{ 'a' in {"a": 1} }},{{ 'a' in {} }}"#), "True,False")
    expectNoDifference(try render(#"{{ 'a' in ["a"] }},{{ 'a' in [] }}"#), "True,False")
    expectNoDifference(try render("{{ 'a' in 'abc' }},{{ 'd' in 'abc' }}"), "True,False")
    expectNoDifference(
      try render("{{ 'a' not in 'abc' }},{{ 'd' not in 'abc' }}"),
      "False,True"
    )
    expectNoDifference(try render("{% set foo = true %}{{ foo is defined }}"), "True")
    expectNoDifference(try render("{% set foo = true %}{{ not foo is defined }}"), "False")
    expectNoDifference(try render("{% set foo = true %}{{ foo is true }}"), "True")
    expectNoDifference(try render("{% set foo = true %}{{ foo is false }}"), "False")
    expectNoDifference(try render("{% set foo = false %}{{ foo is not true }}"), "True")
    expectNoDifference(try render("{% set foo = false %}{{ foo is not false }}"), "False")
    expectNoDifference(try render("{{ 1 is not string }}"), "True")
    expectNoDifference(try render("{{ 'ab' * 3 }}"), "ababab")
    expectNoDifference(try render("{{ [] is iterable }}"), "True")
    expectNoDifference(try render("{{ [] is not number }}"), "True")
    expectNoDifference(try render("{{ not [] }}"), "True")
    expectNoDifference(
      try render(
        "{{ messages[0]['role'] != 'system' }}",
        ["messages": [["role": "system"]]]
      ),
      "False"
    )
    expectNoDifference(
      try render(
        "{{ tool.function.name == 'ipython' }}",
        ["tool": ["function": ["name": "ipython"]]]
      ),
      "True"
    )
    expectNoDifference(try render("{{ (a.b.c) }}", ["a": ["b": ["c": 3]]]), "3")
  }

  @Test
  func `Renders Python And JSON Representations`() throws {
    expectNoDifference(try render(#"{{ {"a": "b"} | tojson }}"#), #"{"a": "b"}"#)
    expectNoDifference(try render(#"{{ {"a": "b"} }}"#), "{'a': 'b'}")
    expectNoDifference(
      try render(
        lines(
          "",
          #"            {%- for x in [1, 1.2, "a", true, True, false, False, None, [], [1], [1, 2], {}, {"a": 1}, {1: "b"}] -%}"#,
          "                {{- x | tojson -}},",
          "            {%- endfor -%}",
          "        "
        )
      ),
      #"1,1.2,"a",true,true,false,false,null,[],[1],[1, 2],{},{"a": 1},{"1": "b"},"#
    )
    expectNoDifference(
      try render("{% set x = [] %}{% set _ = x.append(1) %}{{ x | tojson(indent=2) }}"),
      "[\n  1\n]"
    )
  }

  @Test
  func `Iterates Loops`() throws {
    expectNoDifference(try render(#"{% for x in ["a", "b"] %}{{ x }},{% endfor %}"#), "a,b,")
    expectNoDifference(
      try render(#"{% for x in {"a": 1, "b": 2} %}{{ x }},{% endfor %}"#),
      "a,b,"
    )
    expectNoDifference(try render(#"{% for x in "ab" %}{{ x }},{% endfor %}"#), "a,b,")
    expectNoDifference(
      try render(
        lines(
          "",
          "            {%- for x in range(3) -%}",
          "                {%- if loop.first -%}",
          "                    but first, mojitos!",
          "                {%- endif -%}",
          #"                {{ loop.index }}{{ "," if not loop.last -}}"#,
          "            {%- endfor -%}",
          "        "
        )
      ),
      "but first, mojitos!1,2,3"
    )
    expectNoDifference(try render("{% for i in range(3) %}{{i}},{% endfor %}"), "0,1,2,")
    expectNoDifference(
      try render(
        lines(
          "",
          #"            {%- for x, y in [("a", "b"), ("c", "d")] -%}"#,
          "                {{- x }},{{ y -}};",
          "            {%- endfor -%}",
          "        "
        )
      ),
      "a,b;c,d;"
    )
    expectNoDifference(
      try render(
        lines(
          "",
          "        {%- for x, y in z -%}",
          "            {{- x }},{{ y -}};",
          "        {%- endfor -%}",
          "    "
        ),
        ["z": [[1, 10], [2, 20]]]
      ),
      "1,10;2,20;"
    )
    expectNoDifference(
      try render("{%- for i in range(0) -%}NAH{% else %}OK{% endfor %}"),
      "OK"
    )
    expectNoDifference(
      try render(
        lines(
          "",
          "            {%- for i in range(5) -%}",
          "                ({{ i }}, {{ loop.cycle('odd', 'even') }}),",
          "            {%- endfor -%}",
          "        "
        )
      ),
      "(0, odd),(1, even),(2, odd),(3, even),(4, odd),"
    )
    expectNoDifference(
      try render(
        "{%- for i in range(5) if i % 2 == 0 -%}\n"
          + "{{ i }}, first={{ loop.first }}, last={{ loop.last }}, index={{ loop.index }}, "
          + "index0={{ loop.index0 }}, revindex={{ loop.revindex }}, "
          + "revindex0={{ loop.revindex0 }}, prev={{ loop.previtem }}, "
          + "next={{ loop.nextitem }},\n"
          + "{% endfor -%}"
      ),
      "0, first=True, last=False, index=1, index0=0, revindex=3, revindex0=2, prev=, next=2,\n"
        + "2, first=False, last=False, index=2, index0=1, revindex=2, revindex0=1, prev=0, "
        + "next=4,\n"
        + "4, first=False, last=True, index=3, index0=2, revindex=1, revindex0=0, prev=2, "
        + "next=,\n"
    )
    expectNoDifference(
      try render(
        "{% for i in range(10) %}{{ i }},{% if i == 2 %}{% break %}{% endif %}{% endfor %}"
      ),
      "0,1,2,"
    )
    expectNoDifference(
      try render(
        "{% for i in range(10) %}{% if i % 2 %}{% continue %}{% endif %}{{ i }},{% endfor %}"
      ),
      "0,2,4,6,8,"
    )
  }

  @Test
  func `Calls Collection Methods`() throws {
    expectNoDifference(
      try render("{{ {'a': 1, 'b': 2}.keys() | list }},{{ {}.keys() | list }}"),
      "['a', 'b'],[]"
    )
    expectNoDifference(
      try render(
        lines(
          "",
          "            {%- set o = [0, 1, 2, 3] -%}",
          "            {%- set _ = o.pop() -%}",
          "            {{- o | tojson -}}",
          "            {%- set _ = o.pop(1) -%}",
          "            {{- o | tojson -}}",
          "        "
        )
      ),
      "[0, 1, 2][0, 2]"
    )
    expectNoDifference(
      try render(
        lines(
          "",
          #"            {%- set o = {"x": 1, "y": 2} -%}"#,
          #"            {%- set _ = o.pop("x") -%}"#,
          "            {{- o | tojson -}}",
          "        "
        )
      ),
      #"{"y": 2}"#
    )
    expectNoDifference(
      try render("{{ {1: 2}.get(1) }}; {{ {}.get(1) or '' }}; {{ {}.get(1, 10) }}"),
      "2; ; 10"
    )
    expectNoDifference(
      try render(
        "{% set _ = a.b.append(c.d.e) %}{{ a.b }}",
        ["a": ["b": [1, 2]], "c": ["d": ["e": 3]]]
      ),
      "[1, 2, 3]"
    )
  }

  @Test
  func `Evaluates Subscripts And Slices`() throws {
    expectNoDifference(try render(#"{{ "abcd"[1:-1] }}"#), "bc")
    expectNoDifference(try render("{{ [0, 1, 2, 3][1:-1] }}"), "[1, 2]")
    expectNoDifference(try render("{{ [1, 2, 3][-1] }}"), "3")
    expectNoDifference(
      try render("{% set x = [0, 1, 2, 3] %}{{ x[1:] }}{{ x[:2] }}{{ x[1:3] }}"),
      "[1, 2, 3][0, 1][1, 2]"
    )
    expectNoDifference(
      try render(
        "{% set x = '0123' %}{{ x[1:] }};{{ x[:2] }};{{ x[1:3] }};{{ x[:] }};{{ x[::] }}"
      ),
      "123;01;12;0123;0123"
    )
    expectNoDifference(
      try render(
        "{% set x = [0, 1, 2, 3] %}{{ x[::-1] }}{{ x[:0:-1] }}{{ x[2::-1] }}{{ x[2:0:-1] }}"
          + "{{ x[::2] }}{{ x[::-2] }}{{ x[-2::-2] }}"
      ),
      "[3, 2, 1, 0][3, 2, 1][2, 1, 0][2, 1][0, 2][3, 1][2, 0]"
    )
    expectNoDifference(
      try render(
        "{% set x = '0123' %}{{ x[::-1] }};{{ x[:0:-1] }};{{ x[2::-1] }};{{ x[2:0:-1] }};"
          + "{{ x[::2] }};{{ x[::-2] }};{{ x[-2::-2] }}"
      ),
      "3210;321;210;21;02;31;20"
    )
  }

  @Test
  func `Assigns Variables And Namespaces`() throws {
    expectNoDifference(
      try render(
        "{% set ns = namespace(is_first=false, nottool=false, and_or=true, delme='') %}"
          + "{{ ns.is_first }}"
      ),
      "False"
    )
    expectNoDifference(
      try render("{% set foo %}Hello {{ 'there' }}{% endset %}{{ 1 ~ foo ~ 2 }}"),
      "1Hello there2"
    )
    expectNoDifference(
      try render(
        lines(
          "",
          "            {%- set n = namespace(value=1, title='') -%}",
          #"            {{- n.value }} "{{ n.title }}","#,
          "            {%- set n.value = 2 -%}",
          "            {%- set n.title = 'Hello' -%}",
          #"            {{- n.value }} "{{ n.title }}""#
        )
      ),
      #"1 "",2 "Hello""#
    )
    expectNoDifference(
      try render(
        lines(
          "",
          #"        {%- set user = "Olivier" -%}"#,
          #"        {%- set greeting = "Hello " ~ user -%}"#,
          "        {{- greeting -}}",
          "    "
        )
      ),
      "Hello Olivier"
    )
    expectNoDifference(try render("{% if 1 %}{% elif 1 %}{% else %}{% endif %}"), "")
  }

  @Test
  func `Expands Macros`() throws {
    expectNoDifference(
      try render(
        lines(
          "",
          "            {%- set x = 1 -%}",
          "            {%- set y = 2 -%}",
          "            {%- macro foo(x, z, w=10) -%}",
          "                x={{ x }}, y={{ y }}, z={{ z }}, w={{ w -}}",
          "            {%- endmacro -%}",
          "            {{- foo(100, 3) -}}",
          "        "
        )
      ),
      "x=100, y=2, z=3, w=10"
    )
    expectNoDifference(
      try render(
        lines(
          "",
          "            {% macro input(name, value='', type='text', size=20) -%}",
          #"                <input type="{{ type }}" name="{{ name }}" value="{{ value|e }}" size="{{ size }}">"#,
          "            {%- endmacro -%}",
          "",
          "            <p>{{ input('username') }}</p>",
          "            <p>{{ input('password', type='password') }}</p>"
        )
      ),
      lines(
        "",
        #"            <p><input type="text" name="username" value="" size="20"></p>"#,
        #"            <p><input type="password" name="password" value="" size="20"></p>"#
      )
    )
    expectNoDifference(
      try render(
        lines(
          "",
          "            {#- The values' default array should be created afresh at each call, "
            + "unlike the equivalent Python function -#}",
          "            {%- macro foo(values=[]) -%}",
          "                {%- set _ = values.append(1) -%}",
          "                {{- values -}}",
          "            {%- endmacro -%}",
          "            {{- foo() }} {{ foo() -}}"
        )
      ),
      "[1] [1]"
    )
  }

  @Test
  func `Renders Call Blocks`() throws {
    expectNoDifference(
      try render(
        lines(
          "",
          "            {%- macro test() -%}{{ caller() }},{{ caller() }}{%- endmacro -%}",
          "            {%- call test() -%}x{%- endcall -%}",
          "        "
        )
      ),
      "x,x"
    )
    expectNoDifference(
      try render(
        lines(
          "",
          "            {%- macro outer() -%}Outer[{{ caller() }}]{%- endmacro -%}",
          "            {%- macro inner() -%}Inner({{ caller() }}){%- endmacro -%}",
          "            {%- call outer() -%}{%- call inner() -%}X{%- endcall -%}{%- endcall -%}",
          "        "
        )
      ),
      "Outer[Inner(X)]"
    )
    expectNoDifference(
      try render(
        lines(
          "",
          "            {%- macro test(prefix, suffix) -%}"
            + "{{ prefix }}{{ caller() }}{{ suffix }}{%- endmacro -%}",
          #"            {%- set items = ["a", "b"] -%}"#,
          #"            {%- call test("<ul>", "</ul>") -%}"#,
          "                {%- for item in items -%}",
          "                    <li>{{ item | upper }}</li>",
          "                {%- endfor -%}",
          "            {%- endcall -%}",
          "        "
        )
      ),
      "<ul><li>A</li><li>B</li></ul>"
    )
    expectNoDifference(
      try render(
        lines(
          "",
          "            {%- macro recursive(obj) -%}",
          "            {%- set ns = namespace(content = caller()) -%}",
          "            {%- for key, value in obj.items() %}",
          "                {%- if value is mapping %}",
          "                    {%- call recursive(value) -%}",
          #"                        {{ '\\n\\nclass ' + key.title() + ':\\n' }}"#,
          "                    {%- endcall -%}",
          "                {%- else -%}",
          #"                    {%- set ns.content = ns.content + '  ' + key + ': ' + value + '\\n' -%}"#,
          "                {%- endif -%}",
          "            {%- endfor -%}",
          "            {{ ns.content }}",
          "            {%- endmacro -%}",
          "",
          #"            {%- call recursive({"a": {"b": "1", "c": "2"}}) -%}"#,
          "            {%- endcall -%}",
          "        "
        )
      ),
      "\\n\\nclass A:\\n  b: 1\\n  c: 2\\n"
    )
  }

  @Test
  func `Renders Generation Blocks As Plain Content`() throws {
    expectNoDifference(try render("{% generation %}Foo{% endgeneration %}"), "Foo")
  }

  @Test
  func `Builds Ranges From Arguments`() throws {
    expectNoDifference(
      try render("{{ range(3) | list }}{{ range(4, 7) | list }}{{ range(0, 10, 2) | list }}"),
      "[0, 1, 2][4, 5, 6][0, 2, 4, 6, 8]"
    )
  }

  @Test
  func `Reports Rendering Errors`() {
    expectRenderError("{{ '' | items }}", containing: "Can only get item pairs from a mapping")
    expectRenderError("{{ [] | items }}", containing: "Can only get item pairs from a mapping")
    expectRenderError("{{ None | items }}", containing: "Can only get item pairs from a mapping")
    expectRenderError("{% break %}", containing: "break outside of a loop")
    expectRenderError("{% continue %}", containing: "continue outside of a loop")
    expectRenderError("{%- set _ = [].pop() -%}", containing: "pop from empty list")
    expectRenderError("{%- set _ = {}.pop() -%}", containing: "pop")
    expectRenderError("{%- set _ = {}.pop('foooo') -%}", containing: "foooo")
    expectRenderError(
      "{%- macro test() -%}content{%- endmacro -%}"
        + "{%- call test -%}caller_content{%- endcall -%}",
      containing: "Invalid call block syntax - expected function call"
    )
  }

  @Test
  func `Reports Syntax Errors`() {
    expectRenderError("{% else %}", containing: "Unexpected else")
    expectRenderError("{% endif %}", containing: "Unexpected endif")
    expectRenderError("{% elif 1 %}", containing: "Unexpected elif")
    expectRenderError("{% endfor %}", containing: "Unexpected endfor")
    expectRenderError("{% endfilter %}", containing: "Unexpected endfilter")
    expectRenderError("{% endmacro %}", containing: "Unexpected endmacro")
    expectRenderError("{% endcall %}", containing: "Unexpected endcall")
    expectRenderError("{% if 1 %}", containing: "Unterminated if")
    expectRenderError("{% for x in 1 %}", containing: "Unterminated for")
    expectRenderError("{% generation %}", containing: "Unterminated generation")
    expectRenderError("{% if 1 %}{% else %}", containing: "Unterminated if")
    expectRenderError(
      "{% if 1 %}{% else %}{% elif 1 %}{% endif %}",
      containing: "Unterminated if"
    )
    expectRenderError("{% filter trim %}", containing: "Unterminated filter")
    expectRenderError("{# ", containing: "Missing end of comment tag")
    expectRenderError("{% macro test() %}", containing: "Unterminated macro")
    expectRenderError("{% call test %}", containing: "Unterminated call")
  }
}

private let lstripBlocks = EdgeToolsTemplate.Options(lstripBlocks: true)
private let trimBlocks = EdgeToolsTemplate.Options(trimBlocks: true)
private let lstripTrimBlocks = EdgeToolsTemplate.Options(trimBlocks: true, lstripBlocks: true)

private func expectRenderError(
  _ source: String,
  containing substring: String,
  fileID: StaticString = #fileID,
  filePath: StaticString = #filePath,
  line: UInt = #line,
  column: UInt = #column
) {
  let sourceLocation = SourceLocation(
    fileID: "\(fileID)",
    filePath: "\(filePath)",
    line: Int(line),
    column: Int(column)
  )
  let error = #expect(throws: EdgeToolsTemplateError.self, sourceLocation: sourceLocation) {
    try render(source)
  }
  expectNoDifference(
    error?.message.contains(substring),
    true,
    "\(error?.message ?? "No error") does not contain \(substring)",
    fileID: fileID,
    filePath: filePath,
    line: line,
    column: column
  )
}
