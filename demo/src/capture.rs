use std::env;
use std::fmt::Display;
use std::io::{self, stdout, Write};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, Once,
};
use std::thread::{self, sleep, JoinHandle};
use std::time::Duration;

use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BufferSize, Device, Stream, StreamConfig, SupportedStreamConfigRange};
use crossbeam_channel::{unbounded, Receiver, Sender};
use df::{tract::*, Complex32};
use ndarray::prelude::*;
use ringbuf::{
    traits::{Consumer, Observer, Producer, Split},
    HeapCons, HeapProd, HeapRb,
};
use rubato::{audioadapter_buffers::direct::SequentialSliceOfVecs, Fft, FixedSync, Resampler};

pub type RbProd = HeapProd<f32>;
pub type RbCons = HeapCons<f32>;
pub type SendLsnr = Sender<f32>;
pub type RecvLsnr = Receiver<f32>;
pub type SendSpec = Sender<Box<[f32]>>;
pub type RecvSpec = Receiver<Box<[f32]>>;
pub type SendControl = Sender<(DfControl, f32)>;
pub type RecvControl = Receiver<(DfControl, f32)>;

pub(crate) static INIT_LOGGER: Once = Once::new();
pub(crate) static MODEL_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

const SAMPLE_FORMAT: cpal::SampleFormat = cpal::SampleFormat::F32;

pub struct AudioSink {
    stream: Option<Stream>,
    config: StreamConfig,
    device: Device,
}
pub struct AudioSource {
    stream: Option<Stream>,
    config: StreamConfig,
    device: Device,
}

#[derive(PartialEq)]
pub enum DfControl {
    AttenLim,
    PostFilterBeta,
    MinThreshDb,
    MaxErbThreshDb,
    MaxDfThreshDb,
}

/// Initialize DF model and returns sample rate, frame size, and number of frequency bins
fn init_df_params(model_path: Option<PathBuf>) -> Result<(DfParams, usize, usize, usize)> {
    let df_params = if let Some(path) = model_path {
        DfParams::new(path)?
    } else {
        DfParams::default()
    };
    let (sr, frame_size, freq_size) = df_params.runtime_info()?;
    Ok((df_params, sr, frame_size, freq_size))
}

#[derive(Clone, Copy)]
enum StreamDirection {
    Input,
    Output,
}
impl Display for StreamDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamDirection::Input => write!(f, "input"),
            StreamDirection::Output => write!(f, "output"),
        }
    }
}

fn get_all_configs(device: &Device, direction: StreamDirection) -> Vec<SupportedStreamConfigRange> {
    match direction {
        StreamDirection::Input => device
            .supported_input_configs()
            .expect("Failed to get input configs")
            .collect::<Vec<SupportedStreamConfigRange>>(),
        StreamDirection::Output => device
            .supported_output_configs()
            .expect("Failed to get output configs")
            .collect::<Vec<SupportedStreamConfigRange>>(),
    }
}

fn get_stream_config(
    device: &Device,
    sample_rate: u32,
    direction: StreamDirection,
    frame_size: usize,
) -> Option<StreamConfig> {
    let mut configs = Vec::new();
    let all_configs = get_all_configs(device, direction);
    for c in all_configs.iter() {
        if c.channels() == 1 && c.sample_format() == SAMPLE_FORMAT {
            log::debug!("Found audio {} config: {:?}", direction, &c);
            configs.push(*c);
        }
    }
    // Further add multi-channel configs if no mono was found. The signal will be downmixed later.
    for c in all_configs.iter() {
        if c.channels() >= 2 && c.sample_format() == SAMPLE_FORMAT {
            log::debug!("Found audio source config: {:?}", &c);
            configs.push(*c);
        }
    }
    assert!(
        !configs.is_empty(),
        "No suitable audio {} config found.",
        direction
    );
    let sr = sample_rate;
    for c in configs.iter() {
        if sr >= c.min_sample_rate() && sr <= c.max_sample_rate() {
            let mut c: StreamConfig = (*c).with_sample_rate(sr).into();
            c.buffer_size = BufferSize::Fixed(frame_size as u32);
            return Some(c);
        }
    }

    if let Some(c) = configs.first() {
        let mut c: StreamConfig = (*c).with_max_sample_rate().into();
        c.buffer_size = BufferSize::Fixed(frame_size as u32 * c.sample_rate / sample_rate);
        log::warn!("Using best matching config {:?}", c);
        return Some(c);
    }
    None
}

impl AudioSink {
    fn new(sample_rate: u32, frame_size: usize, device_str: Option<String>) -> Result<Self> {
        let host = cpal::default_host();
        let mut device = host.default_output_device().expect("no output device available");
        if let Some(device_str) = device_str {
            for avail_dev in host.output_devices()? {
                if avail_dev
                    .description()?
                    .name()
                    .to_lowercase()
                    .contains(&device_str.to_lowercase())
                {
                    device = avail_dev
                }
            }
        }
        let config = get_stream_config(&device, sample_rate, StreamDirection::Output, frame_size)
            .expect("No suitable audio output config found.");

        Ok(Self {
            stream: None,
            config,
            device,
        })
    }
    fn start(&mut self, mut rb: RbCons) -> Result<()> {
        let ch = self.config.channels;
        let needs_upmix = ch > 1;
        let stream = self.device.build_output_stream(
            &self.config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let len = data.len() / ch as usize;
                let mut n = 0;
                if needs_upmix {
                    let mut data_it = data.chunks_mut(ch as usize);
                    while n < len {
                        for (i, o) in rb.pop_iter().zip(&mut data_it) {
                            o.fill(i);
                            n += 1;
                        }
                    }
                } else {
                    while n < len {
                        n += rb.pop_slice(&mut data[n..]);
                    }
                }
                debug_assert_eq!(n, len);
                if log::log_enabled!(log::Level::Trace) {
                    log::trace!(
                        "Returning data to audio sink with len: {}, rms: {}",
                        len,
                        df::rms(data.iter())
                    );
                }
            },
            move |err| log::error!("Error during audio output {:?}", err),
            None, // None=blocking, Some(Duration)=timeout
        )?;
        stream.play()?;
        log::info!(
            "Starting playback stream on device {}",
            self.device.description()?.name()
        );
        self.stream = Some(stream);
        Ok(())
    }
    fn sr(&self) -> u32 {
        self.config.sample_rate
    }
    fn pause(&mut self) -> Result<()> {
        if let Some(s) = self.stream.as_mut() {
            s.pause()?;
        }
        Ok(())
    }
}

impl AudioSource {
    fn new(sample_rate: u32, frame_size: usize, device_str: Option<String>) -> Result<Self> {
        let host = cpal::default_host();
        let mut device = host.default_input_device().expect("no output device available");
        if let Some(device_str) = device_str {
            for avail_dev in host.input_devices()? {
                if avail_dev
                    .description()?
                    .name()
                    .to_lowercase()
                    .contains(&device_str.to_lowercase())
                {
                    device = avail_dev
                }
            }
        }
        let config = get_stream_config(&device, sample_rate, StreamDirection::Input, frame_size)
            .expect("No suitable audio input config found.");

        Ok(Self {
            stream: None,
            config,
            device,
        })
    }
    fn start(&mut self, mut rb: RbProd) -> Result<()> {
        let ch = self.config.channels;
        let needs_downmix = ch > 1;
        let stream = self.device.build_input_stream(
            &self.config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let len = data.len() / ch as usize;
                if log::log_enabled!(log::Level::Trace) {
                    log::trace!(
                        "Got data from audio source with len: {}, rms: {}",
                        len,
                        df::rms(data.iter())
                    );
                }
                let mut n = 0;
                if needs_downmix {
                    let mut iter = data.chunks(ch as usize).map(df::mean);
                    while n < len {
                        n += rb.push_iter(&mut iter);
                    }
                } else {
                    while n < len {
                        n += rb.push_slice(&data[n..]);
                    }
                }
                debug_assert_eq!(n, len);
            },
            move |err| log::error!("Error during audio output {:?}", err),
            None, // None=blocking, Some(Duration)=timeout
        )?;
        log::info!(
            "Starting caputre stream on device {}",
            self.device.description()?.name()
        );
        stream.play()?;
        self.stream = Some(stream);
        Ok(())
    }
    fn sr(&self) -> u32 {
        self.config.sample_rate
    }
    fn pause(&mut self) -> Result<()> {
        if let Some(s) = self.stream.as_mut() {
            s.pause()?;
        }
        Ok(())
    }
}

pub(crate) struct AtomicControls {
    has_init: Arc<AtomicBool>,
    should_stop: Arc<AtomicBool>,
}
impl AtomicControls {
    pub fn into_inner(self) -> (Arc<AtomicBool>, Arc<AtomicBool>) {
        (self.has_init, self.should_stop)
    }
}
pub(crate) struct GuiCom {
    pub s_lsnr: Option<SendLsnr>,
    pub s_spec: Option<(SendSpec, SendSpec)>,
    pub r_opt: Option<RecvControl>,
}
impl GuiCom {
    pub fn into_inner(
        self,
    ) -> (
        Option<SendLsnr>,
        Option<(SendSpec, SendSpec)>,
        Option<RecvControl>,
    ) {
        (self.s_lsnr, self.s_spec, self.r_opt)
    }
}

fn get_worker_fn(
    mut rb_in: RbCons,
    mut rb_out: RbProd,
    df_params: DfParams,
    channels: usize,
    sample_rates: (usize, usize),
    controls: AtomicControls,
    df_com: Option<GuiCom>,
) -> (impl FnOnce(), Receiver<std::result::Result<(), String>>) {
    let (ready_tx, ready_rx) = crossbeam_channel::bounded(1);
    let (input_sr, output_sr) = sample_rates;
    let (has_init, should_stop) = controls.into_inner();
    let (mut s_lsnr, mut s_spec, mut r_opt) = if let Some(df_com) = df_com {
        df_com.into_inner()
    } else {
        (None, None, None)
    };
    let worker = move || {
        let r_params = RuntimeParams::default_with_ch(channels);
        let mut df = match DfTract::new(df_params, &r_params) {
            Ok(df) => df,
            Err(error) => {
                let _ = ready_tx.send(Err(format!(
                    "Could not initialize DeepFilter runtime: {error}"
                )));
                return;
            }
        };
        debug_assert_eq!(df.ch, 1); // Processing for more channels are not implemented yet
        let mut inframe = Array2::zeros((df.ch, df.hop_size));
        let mut outframe = inframe.clone();
        if let Err(error) = df.process(inframe.view(), outframe.view_mut()) {
            let _ = ready_tx.send(Err(format!("Failed to run DeepFilterNet: {error}")));
            return;
        }
        has_init.store(true, Ordering::Relaxed);
        let _ = ready_tx.send(Ok(()));
        log::info!("Worker init");
        let mut input_resampler = if input_sr != df.sr {
            let r = Fft::<f32>::new(input_sr, df.sr, df.hop_size, 1, 1, FixedSync::Output)
                .expect("Failed to init input resampler");
            let n_in = r.input_frames_max();
            Some((r, vec![vec![0.; n_in]; 1], vec![vec![0.; df.hop_size]; 1]))
        } else {
            None
        };
        let mut output_resampler = if output_sr != df.sr {
            let r = Fft::<f32>::new(df.sr, output_sr, df.hop_size, 1, 1, FixedSync::Input)
                .expect("Failed to init output resampler");
            let n_out = r.output_frames_max();
            Some((r, vec![vec![0.; df.hop_size]; 1], vec![vec![0.; n_out]; 1]))
        } else {
            None
        };
        while !should_stop.load(Ordering::Relaxed) {
            let n_in = input_resampler
                .as_ref()
                .map(|(r, _, _)| r.input_frames_next())
                .unwrap_or(df.hop_size);
            if rb_in.occupied_len() < n_in {
                // Sleep for half a hop size
                sleep(Duration::from_secs_f32(
                    df.hop_size as f32 / df.sr as f32 / 2.,
                ));
                continue;
            }
            if let Some((ref mut r, ref mut buf, ref mut out)) = input_resampler.as_mut() {
                let n = rb_in.pop_slice(&mut buf[0][..n_in]);
                debug_assert_eq!(n, n_in);
                debug_assert_eq!(n, r.input_frames_next());
                let input = SequentialSliceOfVecs::new(buf, 1, n_in).unwrap();
                let mut output = SequentialSliceOfVecs::new_mut(out, 1, df.hop_size).unwrap();
                let (_, n_out) = r.process_into_buffer(&input, &mut output, None).unwrap();
                debug_assert_eq!(n_out, df.hop_size);
                inframe.as_slice_mut().unwrap().copy_from_slice(&out[0][..df.hop_size]);
            } else {
                let n = rb_in.pop_slice(inframe.as_slice_mut().unwrap());
                debug_assert_eq!(n, n_in);
            }
            let lsnr = df
                .process(inframe.view(), outframe.view_mut())
                .expect("Failed to run DeepFilterNet");
            let mut n = 0;
            if let Some((ref mut r, ref mut buf, ref mut out)) = output_resampler.as_mut() {
                buf[0].copy_from_slice(outframe.as_slice().unwrap());
                let input = SequentialSliceOfVecs::new(buf, 1, df.hop_size).unwrap();
                let mut output =
                    SequentialSliceOfVecs::new_mut(out, 1, r.output_frames_max()).unwrap();
                let (_, n_out) = r.process_into_buffer(&input, &mut output, None).unwrap();
                while n < n_out {
                    n += rb_out.push_slice(&out[0][n..n_out]);
                }
            } else {
                let buf = outframe.as_slice().unwrap();
                let n_out = df.hop_size;
                while n < n_out {
                    n += rb_out.push_slice(&buf[n..]);
                }
            }
            if let Some(ref mut s_lsnr) = s_lsnr.as_mut() {
                s_lsnr.send(lsnr).expect("Failed to send to LSNR rb");
            }
            if let Some((ref mut s_noisy, ref mut s_enh)) = s_spec.as_mut() {
                push_spec(df.get_spec_noisy(), s_noisy);
                push_spec(df.get_spec_enh(), s_enh);
            }
            if let Some(ref mut r_opt) = r_opt.as_mut() {
                while let Ok((c, v)) = r_opt.try_recv() {
                    match c {
                        DfControl::AttenLim => df.set_atten_lim(v),
                        DfControl::PostFilterBeta => df.set_pf_beta(v),
                        DfControl::MinThreshDb => df.min_db_thresh = v,
                        DfControl::MaxErbThreshDb => df.max_db_erb_thresh = v,
                        DfControl::MaxDfThreshDb => df.max_db_df_thresh = v,
                    }
                }
            }
        }
    };
    (worker, ready_rx)
}

fn push_spec(spec: ArrayView2<Complex32>, sender: &SendSpec) {
    debug_assert_eq!(spec.len_of(Axis(0)), 1); // only single channel for now
    let out = spec.iter().map(|x| x.norm_sqr().max(1e-10).log10() * 10.).collect::<Vec<f32>>();
    sender.send(out.into_boxed_slice()).expect("Failed to send spectrogram")
}

pub fn log_format(buf: &mut env_logger::fmt::Formatter, record: &log::Record) -> io::Result<()> {
    let ts = buf.timestamp_millis();
    let module = record.module_path().unwrap_or("").to_string();
    writeln!(
        buf,
        "{} | {} | {} {}",
        ts,
        record.level(),
        module,
        record.args()
    )
}

pub struct DeepFilterCapture {
    pub sr: usize,
    pub frame_size: usize,
    pub freq_size: usize,
    should_stop: Arc<AtomicBool>,
    worker_handle: Option<JoinHandle<()>>,
    source: AudioSource,
    sink: AudioSink,
}

impl Default for DeepFilterCapture {
    fn default() -> Self {
        DeepFilterCapture::new(None, None, None, None, None)
            .expect("Error during DeepFilterCapture initialization")
    }
}
impl DeepFilterCapture {
    pub fn new(
        model_path: Option<PathBuf>,
        s_lsnr: Option<SendLsnr>,
        s_noisy: Option<SendSpec>,
        s_enh: Option<SendSpec>,
        r_opt: Option<RecvControl>,
    ) -> Result<Self> {
        let ch = 1;
        let (df_params, sr, frame_size, freq_size) = init_df_params(model_path)?;
        let in_rb = HeapRb::<f32>::new(frame_size * 100);
        let out_rb = HeapRb::<f32>::new(frame_size * 100);
        let (in_prod, in_cons) = in_rb.split();
        let (out_prod, out_cons) = out_rb.split();
        let mut source = AudioSource::new(sr as u32, frame_size, None)?;
        let mut sink = AudioSink::new(sr as u32, frame_size, None)?;
        let should_stop = Arc::new(AtomicBool::new(false));
        let has_init = Arc::new(AtomicBool::new(false));
        let s_spec = match (s_noisy, s_enh) {
            (Some(n), Some(e)) => Some((n, e)),
            _ => None,
        };
        let controls = AtomicControls {
            has_init: has_init.clone(),
            should_stop: should_stop.clone(),
        };
        let df_com = GuiCom {
            s_lsnr,
            s_spec,
            r_opt,
        };
        let (worker, ready_rx) = get_worker_fn(
            in_cons,
            out_prod,
            df_params,
            ch,
            (source.sr() as usize, sink.sr() as usize),
            controls,
            Some(df_com),
        );
        let worker_handle = Some(thread::spawn(worker));
        match ready_rx.recv_timeout(Duration::from_secs(60)) {
            Ok(Ok(())) => (),
            Ok(Err(error)) => {
                should_stop.store(true, Ordering::Relaxed);
                anyhow::bail!("DeepFilter worker failed to initialize: {error}");
            }
            Err(_) => {
                should_stop.store(true, Ordering::Relaxed);
                anyhow::bail!("DeepFilter worker terminated or timed out during initialization");
            }
        }
        log::info!("DeepFilter Capture init");
        source.start(in_prod)?;
        sink.start(out_cons)?;

        Ok(Self {
            sr,
            frame_size,
            freq_size,
            should_stop,
            worker_handle,
            source,
            sink,
        })
    }

    pub fn should_stop(&mut self) -> Result<()> {
        self.sink.pause()?;
        self.source.pause()?;
        if let Some(h) = self.worker_handle.take() {
            log::info!("Joining DF Worker");
            self.should_stop.swap(true, Ordering::Relaxed);
            if let Err(error) = h.join() {
                log::error!("DF worker thread panicked: {error:?}");
            }
        }
        Ok(())
    }
}

#[allow(unused)]
#[allow(unknown_lints)] // assigning_clones is clippy nightly only
#[allow(clippy::assigning_clones)]
pub fn main() -> Result<()> {
    INIT_LOGGER.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
            .filter_module("tract_onnx", log::LevelFilter::Error)
            .filter_module("tract_core", log::LevelFilter::Error)
            .filter_module("tract_hir", log::LevelFilter::Error)
            .filter_module("tract_linalg", log::LevelFilter::Error)
            .format(log_format)
            .init();
    });

    let (lsnr_prod, mut lsnr_cons) = unbounded();
    let mut model_path = env::var("DF_MODEL").ok().map(PathBuf::from);
    if model_path.is_none() {
        model_path = MODEL_PATH.lock().unwrap().clone();
    }
    if let Some(p) = model_path.as_ref() {
        log::info!("Running with model '{:?}'", p);
    }
    let _c = DeepFilterCapture::new(model_path, Some(lsnr_prod), None, None, None);

    loop {
        sleep(Duration::from_millis(200));
        while let Ok(lsnr) = lsnr_cons.try_recv() {
            print!("\rCurrent SNR: {:>5.1} dB{esc}[1;", lsnr, esc = 27 as char);
        }
        stdout().flush().unwrap();
    }
}
