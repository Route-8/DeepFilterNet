use std::boxed::Box;

use ndarray::prelude::*;
use wasm_bindgen::prelude::*;

use crate::tract::*;

// Initialize panic hook for better error reporting in browser console
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    #[cfg(feature = "wasm")]
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct DFState {
    inner: crate::tract::DfTract,
    input_buf: Array2<f32>,
    output_buf: Array2<f32>,
}

#[wasm_bindgen]
impl DFState {
    fn new(model_bytes: &[u8], channels: usize, atten_lim: f32) -> Self {
        let r_params = RuntimeParams::default_with_ch(channels).with_atten_lim(atten_lim);

        // This will panic with detailed error message (caught by panic hook)
        let df_params = DfParams::from_bytes(model_bytes).expect("Could not load model from bytes");
        let m =
            DfTract::new(df_params, &r_params).expect("Could not initialize DeepFilter runtime");
        let hop_size = m.hop_size;
        DFState {
            inner: m,
            input_buf: Array2::zeros((1, hop_size)),
            output_buf: Array2::zeros((1, hop_size)),
        }
    }
    fn boxed(self) -> Box<DFState> {
        Box::new(self)
    }
}

/// Create a DeepFilterNet Model
///
/// Args:
///     - path: File path to a DeepFilterNet tar.gz onnx model
///     - atten_lim: Attenuation limit in dB.
///
/// Returns:
///     - DF state doing the full processing: stft, DNN noise reduction, istft.
#[wasm_bindgen]
pub unsafe fn df_create(
    model_bytes: &[u8],
    // channels: usize,
    atten_lim: f32,
) -> *mut DFState {
    let df = DFState::new(model_bytes, 1, atten_lim);
    Box::into_raw(df.boxed())
}

/// Destroy a DeepFilterNet state created by [`df_create`].
///
/// Persistent input/output pointers and their typed-array views become invalid
/// immediately. Passing the same non-null pointer more than once is invalid.
#[wasm_bindgen]
pub unsafe fn df_destroy(st: *mut DFState) {
    if !st.is_null() {
        drop(unsafe { Box::from_raw(st) });
    }
}

/// Get DeepFilterNet frame size in samples.
#[wasm_bindgen]
pub unsafe fn df_get_frame_length(st: *mut DFState) -> usize {
    let state = st.as_mut().expect("Invalid pointer");
    state.inner.hop_size
}

/// Set DeepFilterNet attenuation limit.
///
/// Args:
///     - lim_db: New attenuation limit in dB.
#[wasm_bindgen]
pub unsafe fn df_set_atten_lim(st: *mut DFState, lim_db: f32) {
    let state = st.as_mut().expect("Invalid pointer");
    state.inner.set_atten_lim(lim_db)
}

/// Set DeepFilterNet post filter beta. A beta of 0 disables the post filter.
///
/// Args:
///     - beta: Post filter attenuation. Suitable range between 0.05 and 0;
#[wasm_bindgen]
pub unsafe fn df_set_post_filter_beta(st: *mut DFState, beta: f32) {
    let state = st.as_mut().expect("Invalid pointer");
    state.inner.set_pf_beta(beta)
}

/// Processes a chunk of samples.
///
/// Args:
///     - df_state: Created via df_create()
///     - input: Input buffer of length df_get_frame_length()
///     - output: Output buffer of length df_get_frame_length()
///
/// Returns:
///     - Local SNR of the current frame.
#[wasm_bindgen]
pub unsafe fn df_process_frame(st: *mut DFState, input: &[f32]) -> js_sys::Float32Array {
    let state = st.as_mut().expect("Invalid pointer");
    let input = ArrayView2::from_shape((1, state.inner.hop_size), input).unwrap();
    let output_view = state.output_buf.view_mut();
    let _lsnr = state.inner.process(input, output_view).expect("Failed to process DF frame");
    js_sys::Float32Array::from(state.output_buf.as_slice().unwrap())
}

/// Stable byte offset of the persistent input frame in WASM linear memory.
/// The caller must recreate its typed-array view if WASM memory grows and must
/// not use it after calling [`df_destroy`].
#[wasm_bindgen]
pub unsafe fn df_input_frame_ptr(st: *mut DFState) -> u32 {
    let state = st.as_ref().expect("Invalid pointer");
    state.input_buf.as_ptr() as usize as u32
}

/// Stable byte offset of the persistent output frame in WASM linear memory.
/// The caller must recreate its typed-array view if WASM memory grows and must
/// not use it after calling [`df_destroy`].
#[wasm_bindgen]
pub unsafe fn df_output_frame_ptr(st: *mut DFState) -> u32 {
    let state = st.as_ref().expect("Invalid pointer");
    state.output_buf.as_ptr() as usize as u32
}

/// Process the persistent input frame into the persistent output frame without
/// allocating or copying a JavaScript typed array.
#[wasm_bindgen]
pub unsafe fn df_process_frame_persistent(st: *mut DFState) -> f32 {
    let state = st.as_mut().expect("Invalid pointer");
    state
        .inner
        .process(state.input_buf.view(), state.output_buf.view_mut())
        .expect("Failed to process DF frame")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_frame_buffers_remain_stable() {
        let model = include_bytes!("../../models/DeepFilterNet3_onnx.tar.gz");
        let state_ptr = Box::into_raw(Box::new(DFState::new(model, 1, 100.0)));
        let input_ptr = unsafe { df_input_frame_ptr(state_ptr) };
        let output_ptr = unsafe { df_output_frame_ptr(state_ptr) };
        unsafe { (*state_ptr).input_buf.fill(0.01) };

        let lsnr = unsafe { df_process_frame_persistent(state_ptr) };

        assert!(lsnr.is_finite());
        assert_eq!(unsafe { df_input_frame_ptr(state_ptr) }, input_ptr);
        assert_eq!(unsafe { df_output_frame_ptr(state_ptr) }, output_ptr);
        assert!(unsafe { &*state_ptr }.output_buf.iter().all(|sample| sample.is_finite()));
        unsafe { df_destroy(state_ptr) };
    }
}
