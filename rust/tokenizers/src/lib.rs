use std::{
    cell::RefCell,
    error::Error,
    ffi::c_char,
    panic::{catch_unwind, AssertUnwindSafe},
    ptr, slice,
};

use chrono::{
    format::{Item, StrftimeItems},
    DateTime, TimeZone, Utc,
};
use minijinja::{
    value::{Kwargs, Value},
    Environment, ErrorKind,
};
use minijinja_contrib::pycompat;
use tokenizers::Tokenizer;

const MAXIMUM_VOCABULARY_SIZE: usize = 4_194_304;

const NOW_CONTEXT_KEY: &str = "edge_tools_now";

pub const HF_TOKENIZER_SUCCESS: i32 = 0;
pub const HF_TOKENIZER_FAILURE: i32 = 1;
pub const HF_TOKENIZER_INVALID_ARGUMENT: i32 = 2;
pub const HF_TOKENIZER_BUFFER_TOO_SMALL: i32 = 3;

#[repr(C)]
pub struct hf_tokenizer_t {
    tokenizer: Tokenizer,
}

thread_local! {
    static LAST_ERROR: RefCell<Vec<u8>> = RefCell::new(vec![0]);
}

#[unsafe(no_mangle)]
pub extern "C" fn hf_tokenizer_last_error_message() -> *const c_char {
    LAST_ERROR.with_borrow(|message| message.as_ptr().cast())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hf_tokenizer_create(
    tokenizer_json: *const u8,
    tokenizer_json_count: usize,
    tokenizer: *mut *mut hf_tokenizer_t,
) -> i32 {
    unsafe {
        with_boundary(|| {
            let json = input(
                tokenizer_json,
                tokenizer_json_count,
                "Tokenizer JSON is required.",
            )?;
            let handle = output(tokenizer)?;
            let native = Tokenizer::from_bytes(json).map_err(|error| error.to_string())?;
            *handle = Box::into_raw(Box::new(hf_tokenizer_t { tokenizer: native }));
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hf_tokenizer_destroy(tokenizer: *mut hf_tokenizer_t) {
    unsafe {
        if !tokenizer.is_null() {
            drop(Box::from_raw(tokenizer));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hf_tokenizer_encode(
    tokenizer: *const hf_tokenizer_t,
    text: *const u8,
    text_count: usize,
    add_special_tokens: bool,
    token_ids: *mut i32,
    token_ids_capacity: usize,
    token_ids_count: *mut usize,
) -> i32 {
    unsafe {
        with_boundary(|| {
            let text = text_input(text, text_count, "Text is required.")?;
            let encoding = handle(tokenizer)?
                .encode(text, add_special_tokens)
                .map_err(|error| error.to_string())?;
            let ids = encoding
                .get_ids()
                .iter()
                .map(|id| i32::try_from(*id).map_err(|error| error.to_string()))
                .collect::<Result<Vec<_>, String>>()?;
            fill(token_ids, token_ids_capacity, token_ids_count, &ids)
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hf_tokenizer_decode(
    tokenizer: *const hf_tokenizer_t,
    token_ids: *const i32,
    token_ids_count: usize,
    skip_special_tokens: bool,
    text: *mut u8,
    text_capacity: usize,
    text_count: *mut usize,
) -> i32 {
    unsafe {
        with_boundary(|| {
            let token_ids = input(token_ids, token_ids_count, "Token IDs are required.")?
                .iter()
                .map(|id| u32::try_from(*id).map_err(|error| error.to_string()))
                .collect::<Result<Vec<_>, String>>()?;
            let decoded = handle(tokenizer)?
                .decode(&token_ids, skip_special_tokens)
                .map_err(|error| error.to_string())?;
            fill(text, text_capacity, text_count, decoded.as_bytes())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hf_tokenizer_token_to_id(
    tokenizer: *const hf_tokenizer_t,
    token: *const u8,
    token_count: usize,
    token_id: *mut i32,
    found: *mut bool,
) -> i32 {
    unsafe {
        with_boundary(|| {
            let token = text_input(token, token_count, "A token is required.")?;
            let id = handle(tokenizer)?.token_to_id(token);
            *output(found)? = id.is_some();
            if let Some(id) = id {
                *output(token_id)? = i32::try_from(id).map_err(|error| error.to_string())?;
            }
            Ok(())
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hf_tokenizer_id_to_token(
    tokenizer: *const hf_tokenizer_t,
    token_id: i32,
    token: *mut u8,
    token_capacity: usize,
    token_count: *mut usize,
    found: *mut bool,
) -> i32 {
    unsafe {
        with_boundary(|| {
            let tokenizer = handle(tokenizer)?;
            let value = u32::try_from(token_id)
                .ok()
                .and_then(|token_id| tokenizer.id_to_token(token_id));
            *output(found)? = value.is_some();
            fill(
                token,
                token_capacity,
                token_count,
                value.unwrap_or_default().as_bytes(),
            )
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hf_tokenizer_vocabulary(
    tokenizer: *const hf_tokenizer_t,
    tokens: *mut u8,
    tokens_capacity: usize,
    tokens_count: *mut usize,
    lengths: *mut usize,
    lengths_capacity: usize,
    lengths_count: *mut usize,
) -> i32 {
    unsafe {
        with_boundary(|| {
            let entries = vocabulary(handle(tokenizer)?)?;
            let lengths_value = entries.iter().map(String::len).collect::<Vec<_>>();
            let tokens_value = entries.concat();
            fill(
                tokens,
                tokens_capacity,
                tokens_count,
                tokens_value.as_bytes(),
            )
            .and(fill(
                lengths,
                lengths_capacity,
                lengths_count,
                &lengths_value,
            ))
        })
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hf_template_render(
    source: *const u8,
    source_count: usize,
    context_json: *const u8,
    context_json_count: usize,
    text: *mut u8,
    text_capacity: usize,
    text_count: *mut usize,
) -> i32 {
    unsafe {
        with_boundary(|| {
            let source = text_input(source, source_count, "A template source is required.")?;
            let context = text_input(
                context_json,
                context_json_count,
                "A template context is required.",
            )?;
            let rendered = render(source, context)?;
            fill(text, text_capacity, text_count, rendered.as_bytes())
        })
    }
}

enum CallError {
    InvalidArgument(&'static str),
    Failure(String),
    BufferTooSmall,
}

impl From<String> for CallError {
    fn from(message: String) -> Self {
        Self::Failure(message)
    }
}

fn with_boundary(body: impl FnOnce() -> Result<(), CallError>) -> i32 {
    let outcome = catch_unwind(AssertUnwindSafe(body)).unwrap_or_else(|_| {
        Err(CallError::Failure(
            "The native tokenizer encountered an unexpected failure.".to_owned(),
        ))
    });
    match outcome {
        Ok(()) => {
            set_last_error("");
            HF_TOKENIZER_SUCCESS
        }
        Err(CallError::BufferTooSmall) => {
            set_last_error("An output buffer was too small for the result.");
            HF_TOKENIZER_BUFFER_TOO_SMALL
        }
        Err(CallError::InvalidArgument(message)) => {
            set_last_error(message);
            HF_TOKENIZER_INVALID_ARGUMENT
        }
        Err(CallError::Failure(message)) => {
            set_last_error(&message);
            HF_TOKENIZER_FAILURE
        }
    }
}

fn set_last_error(message: &str) {
    let message = message.split('\0').next().unwrap_or_default();
    LAST_ERROR.with_borrow_mut(|last| {
        last.clear();
        last.extend_from_slice(message.as_bytes());
        last.push(0);
    });
}

unsafe fn handle<'a>(tokenizer: *const hf_tokenizer_t) -> Result<&'a Tokenizer, CallError> {
    unsafe {
        tokenizer
            .as_ref()
            .map(|handle| &handle.tokenizer)
            .ok_or(CallError::InvalidArgument("A tokenizer is required."))
    }
}

unsafe fn output<'a, Value>(value: *mut Value) -> Result<&'a mut Value, CallError> {
    unsafe {
        value
            .as_mut()
            .ok_or(CallError::InvalidArgument("An output value is required."))
    }
}

unsafe fn input<'a, Element>(
    data: *const Element,
    count: usize,
    message: &'static str,
) -> Result<&'a [Element], CallError> {
    unsafe {
        match count {
            0 => Ok(&[]),
            _ if data.is_null() => Err(CallError::InvalidArgument(message)),
            _ => Ok(slice::from_raw_parts(data, count)),
        }
    }
}

unsafe fn text_input<'a>(
    data: *const u8,
    count: usize,
    message: &'static str,
) -> Result<&'a str, CallError> {
    unsafe {
        std::str::from_utf8(input(data, count, message)?).map_err(|error| error.to_string().into())
    }
}

unsafe fn fill<Element: Copy>(
    data: *mut Element,
    capacity: usize,
    count: *mut usize,
    values: &[Element],
) -> Result<(), CallError> {
    unsafe {
        *output(count)? = values.len();
        if data.is_null() {
            return Ok(());
        }
        if values.len() > capacity {
            return Err(CallError::BufferTooSmall);
        }
        if !values.is_empty() {
            ptr::copy_nonoverlapping(values.as_ptr(), data, values.len());
        }
        Ok(())
    }
}

fn vocabulary(tokenizer: &Tokenizer) -> Result<Vec<String>, CallError> {
    let vocabulary = tokenizer.get_vocab(true);
    let size = vocabulary
        .values()
        .copied()
        .max()
        .map_or(0, |id| id as usize + 1);
    if size > MAXIMUM_VOCABULARY_SIZE {
        return Err(CallError::Failure(format!(
            "The tokenizer vocabulary spans {size} token IDs, which exceeds the supported maximum \
             of {MAXIMUM_VOCABULARY_SIZE}."
        )));
    }

    let mut entries = vec![None; size];
    for (token, id) in vocabulary {
        entries[id as usize] = Some(token);
    }
    Ok(entries
        .into_iter()
        .enumerate()
        .map(|(id, token)| token.unwrap_or_else(|| format!("<|edge_tokenizer_unused_{id}|>")))
        .collect())
}

fn render(source: &str, context_json: &str) -> Result<String, CallError> {
    let source = generation_blocks_neutralized(source);
    let mut context: serde_json::Value =
        serde_json::from_str(context_json).map_err(|error| error.to_string())?;
    let now = rendering_instant(&mut context)?;

    let mut environment = Environment::new();
    environment.set_trim_blocks(true);
    environment.set_lstrip_blocks(true);
    environment.set_unknown_method_callback(pycompat::unknown_method_callback);
    environment.add_global(
        "raise_exception",
        Value::from_function(|message: String| -> Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(ErrorKind::InvalidOperation, message))
        }),
    );
    environment.add_global(
        "strftime_now",
        Value::from_function(move |format: String| strftime(now, &format)),
    );
    environment.add_filter("tojson", tojson);

    environment
        .template_from_str(&source)
        .and_then(|template| template.render(Value::from_serialize(&context)))
        .map_err(template_failure)
}

fn rendering_instant(context: &mut serde_json::Value) -> Result<DateTime<Utc>, CallError> {
    let pinned = context
        .as_object_mut()
        .and_then(|context| context.remove(NOW_CONTEXT_KEY));
    match pinned {
        None => Ok(Utc::now()),
        Some(value) => value
            .as_i64()
            .and_then(|seconds| Utc.timestamp_opt(seconds, 0).single())
            .ok_or_else(|| {
                CallError::Failure(format!(
                    "`{NOW_CONTEXT_KEY}` must be whole seconds since the Unix epoch."
                ))
            }),
    }
}

fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, minijinja::Error> {
    let options = JSONOptions {
        ensure_ascii: kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false),
        indent: kwargs.get::<Option<usize>>("indent")?,
        sort_keys: kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false),
    };
    kwargs.assert_all_used()?;
    let value = serde_json::to_value(&value).map_err(|error| {
        minijinja::Error::new(ErrorKind::InvalidOperation, "cannot serialize to JSON")
            .with_source(error)
    })?;
    let mut json = String::new();
    write_json(&mut json, &value, options, 0);
    Ok(Value::from(json))
}

#[derive(Clone, Copy)]
struct JSONOptions {
    ensure_ascii: bool,
    indent: Option<usize>,
    sort_keys: bool,
}

fn write_json(json: &mut String, value: &serde_json::Value, options: JSONOptions, depth: usize) {
    match value {
        serde_json::Value::Null => json.push_str("null"),
        serde_json::Value::Bool(value) => json.push_str(if *value { "true" } else { "false" }),
        serde_json::Value::Number(value) => json.push_str(&value.to_string()),
        serde_json::Value::String(value) => write_json_string(json, value, options.ensure_ascii),
        serde_json::Value::Array(values) => write_json_entries(
            json,
            ('[', ']'),
            values.iter().map(|value| (None, value)),
            options,
            depth,
        ),
        serde_json::Value::Object(entries) => {
            let mut entries = entries.iter().collect::<Vec<_>>();
            if options.sort_keys {
                entries.sort_by(|(one, _), (other, _)| one.cmp(other));
            }
            write_json_entries(
                json,
                ('{', '}'),
                entries.into_iter().map(|(key, value)| (Some(key), value)),
                options,
                depth,
            )
        }
    }
}

fn write_json_entries<'a>(
    json: &mut String,
    brackets: (char, char),
    entries: impl Iterator<Item = (Option<&'a String>, &'a serde_json::Value)>,
    options: JSONOptions,
    depth: usize,
) {
    let mut empty = true;
    json.push(brackets.0);
    for (index, (key, value)) in entries.enumerate() {
        if index > 0 {
            json.push_str(if options.indent.is_some() { "," } else { ", " });
        }
        write_json_padding(json, options.indent, depth + 1);
        if let Some(key) = key {
            write_json_string(json, key, options.ensure_ascii);
            json.push_str(": ");
        }
        write_json(json, value, options, depth + 1);
        empty = false;
    }
    if !empty {
        write_json_padding(json, options.indent, depth);
    }
    json.push(brackets.1);
}

fn write_json_padding(json: &mut String, indent: Option<usize>, depth: usize) {
    if let Some(indent) = indent {
        json.push('\n');
        json.extend(std::iter::repeat_n(' ', indent * depth));
    }
}

fn write_json_string(json: &mut String, value: &str, ensure_ascii: bool) {
    json.push('"');
    for character in value.chars() {
        match character {
            '"' => json.push_str("\\\""),
            '\\' => json.push_str("\\\\"),
            '\n' => json.push_str("\\n"),
            '\r' => json.push_str("\\r"),
            '\t' => json.push_str("\\t"),
            '\u{8}' => json.push_str("\\b"),
            '\u{c}' => json.push_str("\\f"),
            _ if (character as u32) < 0x20 => write_unicode_escape(json, character as u16),
            _ if ensure_ascii && !character.is_ascii() => {
                let mut units = [0u16; 2];
                for unit in character.encode_utf16(&mut units) {
                    write_unicode_escape(json, *unit);
                }
            }
            _ => json.push(character),
        }
    }
    json.push('"');
}

fn write_unicode_escape(json: &mut String, unit: u16) {
    json.push_str(&format!("\\u{unit:04x}"));
}

fn strftime(now: DateTime<Utc>, format: &str) -> Result<String, minijinja::Error> {
    let items = StrftimeItems::new(format).collect::<Vec<_>>();
    if items.iter().any(|item| matches!(item, Item::Error)) {
        return Err(minijinja::Error::new(
            ErrorKind::InvalidOperation,
            format!("`{format}` is not a valid strftime format."),
        ));
    }
    Ok(now.format_with_items(items.iter()).to_string())
}

fn template_failure(error: minijinja::Error) -> CallError {
    let mut message = error.to_string();
    let mut cause = error.source();
    while let Some(error) = cause {
        message.push_str(": ");
        message.push_str(&error.to_string());
        cause = error.source();
    }
    CallError::Failure(message)
}

fn generation_blocks_neutralized(source: &str) -> String {
    let mut rewritten = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(start) = rest.find("{%") {
        let Some(length) = rest[start + 2..].find("%}") else {
            break;
        };
        let end = start + 2 + length + 2;
        let inner = &rest[start + 2..end - 2];
        let marker = |character: &char| "-+".contains(*character);
        let opening = inner.chars().next().filter(marker);
        let closing = inner.chars().next_back().filter(marker);
        let body =
            inner.trim_matches(|character: char| character.is_whitespace() || marker(&character));

        let replacement = match body {
            "generation" => Some("if true"),
            "endgeneration" => Some("endif"),
            _ => None,
        };
        rewritten.push_str(&rest[..start]);
        match replacement {
            Some(replacement) => {
                rewritten.push_str("{%");
                rewritten.extend(opening);
                rewritten.push(' ');
                rewritten.push_str(replacement);
                rewritten.push(' ');
                rewritten.extend(closing);
                rewritten.push_str("%}");
            }
            None => rewritten.push_str(&rest[start..end]),
        }
        rest = &rest[end..];
    }
    rewritten.push_str(rest);
    rewritten
}
