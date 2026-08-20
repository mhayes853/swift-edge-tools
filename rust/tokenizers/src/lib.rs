use std::{
    cell::RefCell,
    ffi::c_char,
    panic::{catch_unwind, AssertUnwindSafe},
    ptr, slice,
};

use tokenizers::Tokenizer;

const MAXIMUM_VOCABULARY_SIZE: usize = 4_194_304;

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
