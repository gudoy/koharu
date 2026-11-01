//! Pure Rust llguidance sampler for constrained decoding.
//!
//! Implements a custom `llama_sampler` using the `llguidance` and `toktrie` Rust crates
//! to enforce grammar constraints (JSON schema, regex, Lark, etc.) during token sampling.

use std::ffi::c_void;
use std::sync::{Arc, Mutex, PoisonError};

use llguidance::Matcher;
use toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};

use crate::model::LlamaModel;
use crate::sampling::LlamaSampler;
use crate::token::LlamaToken;

/// Internal state for the llguidance sampler. The matcher is shared so the
/// generation loop can read the tokens the grammar forces next.
struct LlgContext {
    matcher: Arc<Mutex<Matcher>>,
}

impl LlgContext {
    fn matcher(&self) -> std::sync::MutexGuard<'_, Matcher> {
        // A panic cannot unwind out of these C callbacks, so recover the guard.
        self.matcher.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// llama.cpp's own tokenizer, so llguidance can fast-forward the tokens a
/// grammar forces instead of leaving them to one decode step each.
struct LlamaTokEnv {
    trie: TokTrie,
    model: Arc<LlamaModel>,
}

impl TokenizerEnv for LlamaTokEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
        tokenize_bytes(&self.model, bytes)
    }

    fn tokenize_is_canonical(&self) -> bool {
        true
    }
}

fn tokenize_bytes(model: &LlamaModel, bytes: &[u8]) -> Vec<TokenId> {
    let Ok(length) = std::os::raw::c_int::try_from(bytes.len()) else {
        return Vec::new();
    };
    let mut tokens = vec![0; bytes.len() + 1];
    loop {
        let capacity = std::os::raw::c_int::try_from(tokens.len()).unwrap_or(i32::MAX);
        let written = unsafe {
            koharu_llama_sys::llama_tokenize(
                model.vocab_ptr(),
                bytes.as_ptr().cast(),
                length,
                tokens.as_mut_ptr(),
                capacity,
                false,
                false,
            )
        };
        match usize::try_from(written) {
            Ok(written) => {
                tokens.truncate(written);
                return tokens.into_iter().map(i32::cast_unsigned).collect();
            }
            // A negative count is the buffer size llama.cpp needs.
            Err(_) => tokens.resize(written.unsigned_abs() as usize, 0),
        }
    }
}

/// Build a [`toktrie::TokEnv`] from a [`LlamaModel`]'s vocabulary.
///
/// Use this to construct and reuse an `llguidance::ParserFactory` for the model.
/// Building the tokenizer environment walks and detokenizes the entire vocabulary,
/// so callers should do it once per model rather than once per grammar.
///
/// This mirrors the logic in upstream `llguidance.cpp` — for each token:
/// - Try normal detokenize (special=false)
/// - If empty, detokenize with special=true and prefix with 0xFF marker byte
pub fn llguidance_build_tok_env(model: &Arc<LlamaModel>) -> TokEnv {
    let n_vocab = model.n_vocab().cast_unsigned();
    let tok_eos = {
        let eot = unsafe { koharu_llama_sys::llama_vocab_eot(model.vocab_ptr()) };
        if eot == -1 {
            model.token_eos().0.cast_unsigned()
        } else {
            eot.cast_unsigned()
        }
    };
    let info = TokRxInfo::new(n_vocab, tok_eos);
    let mut eog_tokens = vec![tok_eos];

    let mut words = Vec::with_capacity(n_vocab as usize);
    for i in 0..n_vocab.cast_signed() {
        let token = LlamaToken(i);
        if model.is_eog_token(token) && i.cast_unsigned() != tok_eos {
            eog_tokens.push(i.cast_unsigned());
        }
        let bytes = model
            .token_to_piece_bytes(token, 32, false, None)
            .unwrap_or_default();
        if bytes.is_empty() {
            let special_bytes = model
                .token_to_piece_bytes(token, 32, true, None)
                .unwrap_or_default();
            if special_bytes.is_empty() {
                words.push(vec![]);
            } else {
                let mut marked = Vec::with_capacity(special_bytes.len() + 1);
                marked.push(0xFF);
                marked.extend(special_bytes);
                words.push(marked);
            }
        } else {
            words.push(bytes);
        }
    }

    let trie = TokTrie::from(&info, &words).with_eos_tokens(&eog_tokens);
    Arc::new(LlamaTokEnv {
        trie,
        model: Arc::clone(model),
    })
}

// --- extern "C" vtable callbacks ---

unsafe extern "C" fn llg_name(
    _smpl: *const koharu_llama_sys::llama_sampler,
) -> *const std::os::raw::c_char {
    c"llguidance".as_ptr()
}

unsafe extern "C" fn llg_accept(
    smpl: *mut koharu_llama_sys::llama_sampler,
    token: koharu_llama_sys::llama_token,
) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlgContext>() };
    let mut matcher = ctx.matcher();
    if matcher.is_stopped() {
        return;
    }
    let _ = matcher.consume_token(token.cast_unsigned());
}

unsafe extern "C" fn llg_apply(
    smpl: *mut koharu_llama_sys::llama_sampler,
    cur_p: *mut koharu_llama_sys::llama_token_data_array,
) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlgContext>() };
    let cur_p = unsafe { &mut *cur_p };

    let Ok(mask) = ctx.matcher().compute_mask_or_eos() else {
        return;
    };

    let data = unsafe { std::slice::from_raw_parts_mut(cur_p.data, cur_p.size) };
    for item in data.iter_mut() {
        if !mask.is_allowed(item.id.cast_unsigned()) {
            item.logit = f32::NEG_INFINITY;
        }
    }
}

unsafe extern "C" fn llg_reset(smpl: *mut koharu_llama_sys::llama_sampler) {
    let ctx = unsafe { &mut *(*smpl).ctx.cast::<LlgContext>() };
    let _ = ctx.matcher().reset();
}

unsafe extern "C" fn llg_clone(
    smpl: *const koharu_llama_sys::llama_sampler,
) -> *mut koharu_llama_sys::llama_sampler {
    let ctx = unsafe { &*(*smpl).ctx.cast::<LlgContext>() };
    let new_ctx = Box::new(LlgContext {
        matcher: Arc::new(Mutex::new(ctx.matcher().deep_clone())),
    });
    unsafe {
        koharu_llama_sys::llama_sampler_init(
            &raw mut LLG_SAMPLER_I,
            Box::into_raw(new_ctx).cast::<c_void>(),
        )
    }
}

unsafe extern "C" fn llg_free(smpl: *mut koharu_llama_sys::llama_sampler) {
    let ctx_ptr = unsafe { (*smpl).ctx.cast::<LlgContext>() };
    if !ctx_ptr.is_null() {
        drop(unsafe { Box::from_raw(ctx_ptr) });
    }
}

static mut LLG_SAMPLER_I: koharu_llama_sys::llama_sampler_i = koharu_llama_sys::llama_sampler_i {
    name: Some(llg_name),
    accept: Some(llg_accept),
    apply: Some(llg_apply),
    reset: Some(llg_reset),
    clone: Some(llg_clone),
    free: Some(llg_free),
    backend_init: None,
    backend_accept: None,
    backend_apply: None,
    backend_set_input: None,
    backend_reset: None,
    copy_state: None,
};

/// Wrap an already-built [`llguidance::Matcher`] in a llama.cpp sampler.
impl From<Matcher> for LlamaSampler {
    fn from(matcher: Matcher) -> Self {
        Self::from(Arc::new(Mutex::new(matcher)))
    }
}

/// Wrap a [`llguidance::Matcher`] the caller keeps a handle to, for example to
/// read the tokens the grammar forces after each accepted token.
impl From<Arc<Mutex<Matcher>>> for LlamaSampler {
    fn from(matcher: Arc<Mutex<Matcher>>) -> Self {
        let ctx = Box::new(LlgContext { matcher });
        let sampler = unsafe {
            koharu_llama_sys::llama_sampler_init(
                &raw mut LLG_SAMPLER_I,
                Box::into_raw(ctx).cast::<c_void>(),
            )
        };
        LlamaSampler { sampler }
    }
}
