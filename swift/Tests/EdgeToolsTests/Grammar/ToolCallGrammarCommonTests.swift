#if XGrammar
  import CustomDump
  import EdgeTools
  import Testing

  @Suite
  struct `ToolCallGrammarCommon tests`: ~Copyable {
    private let compiler: XGRCompiler
    private let tokenizer: NeedleSPTokenizer
    private let eosToken: EdgeToolsToken.ID

    init() throws {
      let tokenizer = try testTokenizer()
      let compiler = try makeGenericXGRCompiler(tokenizer: tokenizer)
      let eosToken = try requiredTestEOSToken(tokenizer: tokenizer)
      self.tokenizer = tokenizer
      self.compiler = compiler
      self.eosToken = eosToken
    }

    @Test
    func `All Grammars Accept An Empty Tool Call Collection`() throws {
      for fixture in toolCallGrammarTestFixtures {
        let matcher = try self.compiler.makeMatcher(
          try fixture.makeGrammar([.getWeather], .exact(0))
        )
        assertGrammarAccepts(
          fixture.emptyCall,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }
    }

    @Test
    func `All Grammars Accept A Simple Tool Call`() throws {
      for fixture in toolCallGrammarTestFixtures {
        let matcher = try self.compiler.makeMatcher(
          try fixture.makeGrammar([.getWeather], .exact(1))
        )
        assertGrammarAccepts(
          fixture.simpleCall,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }
    }

    @Test
    func `All Grammars Accept The Complex Tool`() throws {
      for fixture in toolCallGrammarTestFixtures {
        let matcher = try self.compiler.makeMatcher(
          try fixture.makeGrammar([.complexTool], .exact(1))
        )
        assertGrammarAccepts(
          fixture.complexCall,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }
    }

    @Test
    func `All Grammars Reject Unknown Tools And Wrong Argument Types`() throws {
      for fixture in toolCallGrammarTestFixtures {
        let unknownToolMatcher = try self.compiler.makeMatcher(
          try fixture.makeGrammar([.getWeather], .exact(1))
        )
        assertGrammarRejects(
          fixture.unknownToolCall,
          matcher: unknownToolMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )

        let wrongTypeMatcher = try self.compiler.makeMatcher(
          try fixture.makeGrammar([.integerTool], .exact(1))
        )
        assertGrammarRejects(
          fixture.wrongTypeCall,
          matcher: wrongTypeMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }
    }

    @Test
    func `All Grammar Fixtures Are Accepted By Their Parsers`() {
      for fixture in toolCallGrammarTestFixtures {
        let calls = parseToolCalls([fixture.complexCall], using: fixture.makeParser)

        expectNoDifference(calls.first?.name, fixture.expectedComplexName)
      }
    }

    @Test
    func `All Grammars Accept Tools Whose Parameters Are Named After Grammar Rules`() throws {
      for fixture in toolCallGrammarTestFixtures {
        let matcher = try self.compiler.makeMatcher(
          try fixture.makeGrammar([.ruleNamedTool], .exact(1))
        )
        assertGrammarAccepts(
          fixture.ruleNamedCall,
          matcher: matcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }
    }

    @Test
    func `All Grammars Enforce Tool Call Ranges`() throws {
      for fixture in toolCallGrammarTestFixtures {
        let acceptingMatcher = try self.compiler.makeMatcher(
          try fixture.makeGrammar([.getWeather], .exact(2))
        )
        assertGrammarAccepts(
          fixture.twoCalls,
          matcher: acceptingMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )

        let rejectingMatcher = try self.compiler.makeMatcher(
          try fixture.makeGrammar([.getWeather], .exact(1))
        )
        assertGrammarRejects(
          fixture.twoCalls,
          matcher: rejectingMatcher,
          tokenizer: self.tokenizer,
          eosToken: self.eosToken
        )
      }
    }
  }

  struct ToolCallGrammarTestFixture: Sendable, CustomStringConvertible {
    let name: String
    let makeGrammar: @Sendable ([EdgeToolDefinition], GrammarToolCallRange) throws -> XGRGrammar
    let makeParser: @Sendable () -> any EdgeToolCallParser
    let expectedComplexName: String
    let emptyCall: String
    let simpleCall: String
    let twoCalls: String
    let unknownToolCall: String
    let wrongTypeCall: String
    let complexCall: String
    let ruleNamedCall: String

    var description: String { self.name }
  }

  let toolCallGrammarTestFixtures = [
    ToolCallGrammarTestFixture(
      name: "Needle",
      makeGrammar: { try XGRGrammar.needle(tools: $0, range: $1) },
      makeParser: { NeedleToolCallParser() },
      expectedComplexName: "complex_tool",
      emptyCall: "<tool_call> []",
      simpleCall: #"<tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}}]"#,
      twoCalls:
        """
        <tool_call> [{"name":"get_weather","arguments":{"location":"Seoul"}},{"name":\
        "get_weather","arguments":{"location":"Paris"}}]
        """,
      unknownToolCall: #"<tool_call> [{"name":"unknown","arguments":{"location":"Seoul"}}]"#,
      wrongTypeCall: #"<tool_call> [{"name":"integer_tool","arguments":{"value":"oops"}}]"#,
      complexCall:
        """
        <tool_call> [{"name":"complex_tool","arguments":{"title":"alpha","count":3.5,"enabled":true,\
        "mode":"execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{\
        "ALPHA":1,"BETA_LABEL":2},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":[\
        "a","b"],"config":{"threshold":0.75,"flags":[true,false]}}}]
        """,
      ruleNamedCall:
        #"<tool_call> [{"name":"rule_named_tool","arguments":{"root":"a","xml_object":"b"}}]"#
    ),
    ToolCallGrammarTestFixture(
      name: "Qwen JSON",
      makeGrammar: { try XGRGrammar.qwenJSON(tools: $0, range: $1) },
      makeParser: { QwenJSONToolCallParser() },
      expectedComplexName: "complexTool",
      emptyCall: "",
      simpleCall: #"<tool_call>{"name":"getWeather","arguments":{"location":"Seoul"}}</tool_call>"#,
      twoCalls:
        """
        <tool_call>{"name":"getWeather","arguments":{"location":"Seoul"}}</tool_call>\
        <tool_call>{"name":"getWeather","arguments":{"location":"Paris"}}</tool_call>
        """,
      unknownToolCall:
        #"<tool_call>{"name":"unknown","arguments":{"location":"Seoul"}}</tool_call>"#,
      wrongTypeCall: #"<tool_call>{"name":"integerTool","arguments":{"value":"oops"}}</tool_call>"#,
      complexCall:
        """
        <tool_call>{"name":"complexTool","arguments":{"title":"alpha","count":3.5,"enabled":true,"mode":\
        "execute","ticket_id":"ABC-12","priority":4,"routing":{"region":"us-west"},"labels":{"ALPHA":1,\
        "BETA_LABEL":2},"window":3,"tuple_args":["alpha",2,true],"optional_note":null,"tags":["a","b"],\
        "config":{"threshold":0.75,"flags":[true,false]}}}</tool_call>
        """,
      ruleNamedCall:
        """
        <tool_call>{"name":"ruleNamedTool","arguments":{"root":"a","xml_object":"b"}}\
        </tool_call>
        """
    ),
    ToolCallGrammarTestFixture(
      name: "Qwen XML",
      makeGrammar: { try XGRGrammar.qwenXML(tools: $0, range: $1) },
      makeParser: { QwenXMLToolCallParser() },
      expectedComplexName: "complexTool",
      emptyCall: "",
      simpleCall:
        "<tool_call><function=getWeather><parameter=location>Seoul</parameter></function></tool_call>",
      twoCalls:
        """
        <tool_call><function=getWeather><parameter=location>Seoul</parameter></function></tool_call>\
        <tool_call><function=getWeather><parameter=location>Paris</parameter></function></tool_call>
        """,
      unknownToolCall:
        "<tool_call><function=unknown><parameter=location>Seoul</parameter></function></tool_call>",
      wrongTypeCall:
        "<tool_call><function=integerTool><parameter=value>oops</parameter></function></tool_call>",
      complexCall:
        """
        <tool_call><function=complexTool><parameter=title>alpha</parameter>\
        <parameter=count>3.5</parameter><parameter=enabled>true</parameter>\
        <parameter=mode>execute</parameter><parameter=ticket_id>ABC-12</parameter>\
        <parameter=priority>4</parameter><parameter=routing>{"region":"us-west"}</parameter>\
        <parameter=labels>{"ALPHA":1,"BETA_LABEL":2}</parameter>\
        <parameter=window>3</parameter><parameter=tuple_args>["alpha",2,true]</parameter>\
        <parameter=optional_note>null</parameter><parameter=tags>["a","b"]</parameter>\
        <parameter=config>{"threshold":0.75,"flags":[true,false]}</parameter></function></tool_call>
        """,
      ruleNamedCall:
        """
        <tool_call><function=ruleNamedTool><parameter=root>a</parameter>\
        <parameter=xml_object>b</parameter></function></tool_call>
        """
    ),
    ToolCallGrammarTestFixture(
      name: "FunctionGemma",
      makeGrammar: { try XGRGrammar.functionGemma(tools: $0, range: $1) },
      makeParser: { FunctionGemmaToolCallParser() },
      expectedComplexName: "complexTool",
      emptyCall: "",
      simpleCall:
        "<start_function_call>call:getWeather{location:<escape>Seoul<escape>}<end_function_call>",
      twoCalls:
        """
        <start_function_call>call:getWeather{location:<escape>Seoul<escape>}<end_function_call>\
        <start_function_call>call:getWeather{location:<escape>Paris<escape>}<end_function_call>
        """,
      unknownToolCall:
        "<start_function_call>call:unknown{location:<escape>Seoul<escape>}<end_function_call>",
      wrongTypeCall:
        "<start_function_call>call:integerTool{value:<escape>oops<escape>}<end_function_call>",
      complexCall:
        """
        <start_function_call>call:complexTool{title:<escape>alpha<escape>,count:<escape>3.5<escape>,\
        enabled:<escape>true<escape>,mode:<escape>execute<escape>,ticket_id:<escape>ABC-12<escape>,\
        priority:<escape>4<escape>,routing:<escape>{"region":"us-west"}<escape>,\
        labels:<escape>{"ALPHA":1,"BETA_LABEL":2}<escape>,window:<escape>3<escape>,\
        tuple_args:<escape>["alpha",2,true]<escape>,optional_note:<escape>null<escape>,\
        tags:<escape>["a","b"]<escape>,\
        config:<escape>{"threshold":0.75,"flags":[true,false]}<escape>}<end_function_call>
        """,
      ruleNamedCall:
        """
        <start_function_call>call:ruleNamedTool{root:<escape>a<escape>,\
        xml_object:<escape>b<escape>}<end_function_call>
        """
    ),
    ToolCallGrammarTestFixture(
      name: "Gemma 4",
      makeGrammar: { try XGRGrammar.gemma4(tools: $0, range: $1) },
      makeParser: { Gemma4ToolCallParser() },
      expectedComplexName: "complexTool",
      emptyCall: "",
      simpleCall:
        "<|tool_call>call:getWeather{location:<|\"|>Seoul<|\"|>}<tool_call|>",
      twoCalls:
        """
        <|tool_call>call:getWeather{location:<|"|>Seoul<|"|>}<tool_call|>\
        <|tool_call>call:getWeather{location:<|"|>Paris<|"|>}<tool_call|>
        """,
      unknownToolCall:
        "<|tool_call>call:unknown{location:<|\"|>Seoul<|\"|>}<tool_call|>",
      wrongTypeCall:
        "<|tool_call>call:integerTool{value:<|\"|>oops<|\"|>}<tool_call|>",
      complexCall:
        """
        <|tool_call>call:complexTool{title:<|"|>alpha<|"|>,count:3.5,enabled:true,mode:\
        execute,ticket_id:ABC-12,priority:4,routing:{"region":"us-west"},\
        labels:{"ALPHA":1,"BETA_LABEL":2},window:3,tuple_args:["alpha",2,true],\
        optional_note:null,tags:["a","b"],config:{"threshold":0.75,"flags":[true,false]}}\
        <tool_call|>
        """,
      ruleNamedCall:
        """
        <|tool_call>call:ruleNamedTool{root:<|"|>a<|"|>,xml_object:<|"|>b<|"|>}\
        <tool_call|>
        """
    ),
    ToolCallGrammarTestFixture(
      name: "LFM2P5 Python",
      makeGrammar: { try XGRGrammar.lfm2P5Python(tools: $0, range: $1) },
      makeParser: { LFM2P5PythonToolCallParser() },
      expectedComplexName: "complexTool",
      emptyCall: "<|tool_call_start|>[]<|tool_call_end|>",
      simpleCall: #"<|tool_call_start|>[getWeather(location="Seoul")]<|tool_call_end|>"#,
      twoCalls:
        """
        <|tool_call_start|>[getWeather(location="Seoul"),\
        getWeather(location="Paris")]<|tool_call_end|>
        """,
      unknownToolCall: #"<|tool_call_start|>[unknown(location="Seoul")]<|tool_call_end|>"#,
      wrongTypeCall: #"<|tool_call_start|>[integerTool(value="oops")]<|tool_call_end|>"#,
      complexCall:
        """
        <|tool_call_start|>[complexTool(title="alpha",count=3.5,enabled=True,mode="execute",\
        ticket_id="ABC-12",priority=4,routing={"region":"us-west"},\
        labels={"ALPHA":1,"BETA_LABEL":2},window=3,tuple_args=["alpha",2,True],\
        optional_note=None,tags=["a","b"],config={"threshold":0.75,"flags":[True,False]})]\
        <|tool_call_end|>
        """,
      ruleNamedCall:
        #"<|tool_call_start|>[ruleNamedTool(root="a",xml_object="b")]<|tool_call_end|>"#
    ),
    ToolCallGrammarTestFixture(
      name: "MiniCPM5",
      makeGrammar: { try XGRGrammar.miniCPM5(tools: $0, range: $1) },
      makeParser: { MiniCPM5ToolCallParser() },
      expectedComplexName: "complexTool",
      emptyCall: "",
      simpleCall:
        #"<function name="getWeather"><param name="location">"Seoul"</param></function>"#,
      twoCalls:
        """
        <function name="getWeather"><param name="location">"Seoul"</param></function>
        <function name="getWeather"><param name="location">"Paris"</param></function>
        """,
      unknownToolCall:
        #"<function name="unknown"><param name="location">"Seoul"</param></function>"#,
      wrongTypeCall:
        #"<function name="integerTool"><param name="value">"oops"</param></function>"#,
      complexCall:
        """
        <function name="complexTool"><param name="title">"alpha"</param>\
        <param name="count">3.5</param><param name="enabled">true</param>\
        <param name="mode">"execute"</param><param name="ticket_id">"ABC-12"</param>\
        <param name="priority">4</param><param name="routing">{"region":"us-west"}</param>\
        <param name="labels">{"ALPHA":1,"BETA_LABEL":2}</param><param name="window">3</param>\
        <param name="tuple_args">["alpha",2,true]</param><param name="optional_note">null</param>\
        <param name="tags">["a","b"]</param>\
        <param name="config">{"threshold":0.75,"flags":[true,false]}</param></function>
        """,
      ruleNamedCall:
        #"<function name="ruleNamedTool"><param name="root">"a"</param><param name="xml_object">"b"</param></function>"#
    )
  ]
#endif
