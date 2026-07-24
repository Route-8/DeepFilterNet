use std::collections::VecDeque;
use std::fmt;
use std::io::{self, Write};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender, TrySendError},
    Arc, Mutex, Once, OnceLock,
};
use std::thread::{self, sleep, JoinHandle};
use std::time::{Duration, Instant};

use df::tract::*;
use ladspa::{DefaultValue, Plugin, PluginDescriptor, Port, PortConnection, PortDescriptor};
use ndarray::prelude::*;
use uuid::Uuid;

static INIT_LOGGER: Once = Once::new();

type SampleQueue = Arc<Mutex<Vec<VecDeque<f32>>>>;
type ControlProd = SyncSender<(DfControl, f32)>;
type ControlRecv = Receiver<(DfControl, f32)>;
#[cfg(feature = "dbus")]
use ::{
    event_listener::{Event, Listener},
    zbus::{
        blocking::connection::Builder as ConnectionBuilder, interface, object_server::Interface,
    },
};
#[cfg(feature = "dbus")]
const DBUS_NAME: &str = "org.deepfilter.DeepFilterLadspa";
#[cfg(feature = "dbus")]
const DBUS_PATH: &str = "/org/deepfilter/DeepFilterLadspa";

const ATTEN_LIM_DEF: DefaultValue = DefaultValue::Maximum;
const ATTEN_LIM_MIN: f32 = 0.;
const ATTEN_LIM_MAX: f32 = 100.;
const PF_BETA_DEF: DefaultValue = DefaultValue::Minimum;
const PF_BETA_MIN: f32 = 0.;
const PF_BETA_MAX: f32 = 0.05;
const MIN_PROC_THRESH_DEF: DefaultValue = DefaultValue::Minimum;
const MIN_PROC_THRESH_MIN: f32 = -15.;
const MIN_PROC_THRESH_MAX: f32 = 35.;
const MAX_ERB_BUF_DEF: DefaultValue = DefaultValue::Maximum;
const MAX_ERB_BUF_MIN: f32 = -15.;
const MAX_ERB_BUF_MAX: f32 = 35.;
const MAX_DF_BUF_DEF: DefaultValue = DefaultValue::Maximum;
const MAX_DF_BUF_MIN: f32 = -15.;
const MAX_DF_BUF_MAX: f32 = 35.;
const MIN_PROC_BUF_DEF: DefaultValue = DefaultValue::Minimum;
const MIN_PROC_BUF_MIN: f32 = 0.;
const MIN_PROC_BUF_MAX: f32 = 10.;

struct DfPlugin {
    i_tx: SampleQueue,
    o_rx: SampleQueue,
    control_tx: ControlProd,
    id: String,
    ch: usize,
    sr: usize,
    frame_size: usize,
    proc_delay: usize,
    t_proc_change: usize,
    sleep_duration: Duration,
    control_hist: DfControlHistory,
    cancelled: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    worker_failed: bool,
    #[cfg(feature = "dbus")]
    dbus: Option<(JoinHandle<()>, Arc<Event>, Arc<AtomicBool>)>,
}

const ID_MONO: u64 = 7843795;
const ID_STEREO: u64 = 7843796;
const WORKER_INIT_TIMEOUT: Duration = Duration::from_secs(60);
const WORKER_STALL_FRAMES: u32 = 10;
#[cfg(feature = "dbus")]
const DBUS_INIT_TIMEOUT: Duration = Duration::from_secs(5);
static DF_PARAMS: OnceLock<DfParams> = OnceLock::new();

fn log_format(buf: &mut env_logger::fmt::Formatter, record: &log::Record) -> io::Result<()> {
    let ts = buf.timestamp_millis();
    let module = if let Some(m) = record.module_path() {
        format!(" {} |", m.replace("::reexport_dataset_modules:", ""))
    } else {
        "".to_string()
    };
    writeln!(
        buf,
        "{} | {} | {} {}",
        ts,
        record.level(),
        module,
        record.args()
    )
}

fn syslog_format(buf: &mut env_logger::fmt::Formatter, record: &log::Record) -> io::Result<()> {
    writeln!(
        buf,
        "<{}>{}: {}",
        match record.level() {
            log::Level::Error => 3,
            log::Level::Warn => 4,
            log::Level::Info => 6,
            log::Level::Debug => 7,
            log::Level::Trace => 7,
        },
        record.target(),
        record.args()
    )
}

#[allow(clippy::too_many_arguments)]
fn get_worker_fn(
    inqueue: SampleQueue,
    outqueue: SampleQueue,
    df_params: DfParams,
    channels: usize,
    controls: ControlRecv,
    sleep_duration: Duration,
    id: String,
    cancelled: Arc<AtomicBool>,
) -> (impl FnOnce(), Receiver<Result<(), String>>) {
    let (ready_tx, ready_rx) = sync_channel(1);
    let worker = move || {
        let r_params = RuntimeParams::default_with_ch(channels);
        let mut df = match DfTract::new(df_params, &r_params) {
            Ok(df) => df,
            Err(error) => {
                let error = error.to_string();
                log::error!("DF {id} | Could not initialize DeepFilter runtime: {error}");
                let _ = ready_tx.send(Err(error));
                return;
            }
        };
        if cancelled.load(Ordering::Acquire) {
            return;
        }
        let mut inframe = Array2::zeros((df.ch, df.hop_size));
        let mut outframe = Array2::zeros((df.ch, df.hop_size));
        let t_audio_ms = df.hop_size as f32 / df.sr as f32 * 1000.;
        if ready_tx.send(Ok(())).is_err() {
            return;
        }
        loop {
            if cancelled.load(Ordering::Relaxed) {
                return;
            }
            if let Ok((c, v)) = controls.try_recv() {
                log::info!("DF {} | Setting '{}' to {:.1}", id, c, v);
                match c {
                    DfControl::AttenLim => df.set_atten_lim(v),
                    DfControl::PfBeta => df.set_pf_beta(v),
                    DfControl::MinThreshDb => df.min_db_thresh = v,
                    DfControl::MaxErbThreshDb => df.max_db_erb_thresh = v,
                    DfControl::MaxDfThreshDb => df.max_db_df_thresh = v,
                    _ => (),
                }
            }
            let got_samples = {
                let mut q = inqueue.lock().unwrap();
                if q[0].len() >= df.hop_size {
                    for (i_q_ch, mut i_ch) in q.iter_mut().zip(inframe.outer_iter_mut()) {
                        for i in i_ch.iter_mut() {
                            *i = i_q_ch.pop_front().unwrap();
                        }
                    }
                    true
                } else {
                    false
                }
            };
            if !got_samples {
                sleep(sleep_duration);
                continue;
            }
            let t0 = Instant::now();
            let lsnr = df
                .process(inframe.view(), outframe.view_mut())
                .expect("Error during df::process");
            {
                let mut o_q = outqueue.lock().unwrap();
                for (o_ch, o_q_ch) in outframe.outer_iter().zip(o_q.iter_mut()) {
                    for &o in o_ch.iter() {
                        o_q_ch.push_back(o)
                    }
                }
            }
            let td_ms = t0.elapsed().as_secs_f32() * 1000.;
            log::debug!(
                "DF {} | Enhanced {:.1}ms frame. SNR: {:>5.1}, Processing time: {:>4.1}ms, RTF: {:.2}",
                id,
                t_audio_ms,
                lsnr,
                td_ms,
                td_ms / t_audio_ms
            );
        }
    };
    (worker, ready_rx)
}

/// Initialize DF model and returns sample rate and frame size
fn init_df_params() -> (DfParams, usize, usize) {
    let df_params = DF_PARAMS.get_or_init(DfParams::default).clone();
    let (sr, frame_size, _) =
        df_params.runtime_info().expect("Could not read DeepFilter model runtime info");
    (df_params, sr, frame_size)
}

fn wait_for_worker_ready(
    ready: &Receiver<Result<(), String>>,
    timeout: Duration,
) -> Result<(), String> {
    match ready.recv_timeout(timeout) {
        Ok(result) => result,
        Err(RecvTimeoutError::Disconnected) => {
            Err("DeepFilter worker terminated before initialization completed".to_string())
        }
        Err(RecvTimeoutError::Timeout) => Err(format!(
            "DeepFilter worker initialization timed out after {:.1}s",
            timeout.as_secs_f32()
        )),
    }
}

fn get_new_df(channels: usize) -> impl Fn(&PluginDescriptor, u64) -> DfPlugin {
    move |_: &PluginDescriptor, sample_rate: u64| {
        let t0 = Instant::now();
        let f = match std::env::var("RUST_LOG_STYLE") {
            Ok(s) if s == "SYSTEMD" => syslog_format,
            _ => log_format,
        };
        INIT_LOGGER.call_once(|| {
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
                .filter_module("polling", log::LevelFilter::Error)
                .filter_module("async_io", log::LevelFilter::Error)
                .filter_module("tract_onnx", log::LevelFilter::Error)
                .filter_module("tract_core", log::LevelFilter::Error)
                .filter_module("tract_hir", log::LevelFilter::Error)
                .filter_module("tract_linalg", log::LevelFilter::Error)
                .format(f)
                .init();
        });

        let (df_params, m_sr, hop) = init_df_params();
        assert_eq!(m_sr as u64, sample_rate, "Unsupported sample rate");
        let i_tx = Arc::new(Mutex::new(vec![VecDeque::with_capacity(hop * 4); channels]));
        let o_rx = Arc::new(Mutex::new(vec![VecDeque::with_capacity(hop * 4); channels]));
        let frame_size = hop;
        let proc_delay = hop;
        // Add a buffer of 1 frame to compensate processing delays causing underruns
        for o_ch in o_rx.lock().unwrap().iter_mut() {
            for _ in 0..proc_delay {
                o_ch.push_back(0f32)
            }
        }
        let sleep_duration = Duration::from_secs_f32(hop as f32 / m_sr as f32 / 5.);
        let id = Uuid::new_v4().as_urn().to_string().split_at(33).1.to_string();

        let (control_tx, control_rx) = sync_channel(32);
        let cancelled = Arc::new(AtomicBool::new(false));

        let (worker, ready_rx) = get_worker_fn(
            Arc::clone(&i_tx),
            Arc::clone(&o_rx),
            df_params,
            channels,
            control_rx,
            sleep_duration,
            id.clone(),
            Arc::clone(&cancelled),
        );
        let worker_handle = thread::spawn(worker);
        let worker_failed =
            if let Err(error) = wait_for_worker_ready(&ready_rx, WORKER_INIT_TIMEOUT) {
                cancelled.store(true, Ordering::Release);
                log::error!("DF {id} | Could not initialize DeepFilter runtime: {error}");
                true
            } else {
                false
            };
        let hist = DfControlHistory::default();
        if !worker_failed {
            log::info!(
                "DF {} | Initialized plugin in {:.1}ms",
                &id,
                t0.elapsed().as_secs_f32() * 1000.
            );
        }
        DfPlugin {
            i_tx,
            o_rx,
            control_tx,
            ch: channels,
            sr: m_sr,
            id,
            frame_size,
            proc_delay,
            t_proc_change: 0,
            sleep_duration,
            control_hist: hist,
            cancelled,
            worker: Some(worker_handle),
            worker_failed,
            #[cfg(feature = "dbus")]
            dbus: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum DfControl {
    AttenLim,
    PfBeta,
    MinThreshDb,
    MaxErbThreshDb,
    MaxDfThreshDb,
    MinBufferFrames,
}
impl DfControl {
    fn from_port_name(name: &str) -> Self {
        match name {
            "Attenuation Limit (dB)" => Self::AttenLim,
            "Post Filter Beta" => Self::PfBeta,
            "Min processing threshold (dB)" => Self::MinThreshDb,
            "Max ERB processing threshold (dB)" => Self::MaxErbThreshDb,
            "Max DF processing threshold (dB)" => Self::MaxDfThreshDb,
            "Min Processing Buffer (frames)" => Self::MinBufferFrames,
            _ => panic!("name not found"),
        }
    }
}
impl fmt::Display for DfControl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DfControl::AttenLim => write!(f, "Attenuation Limit (dB)"),
            DfControl::PfBeta => write!(f, "Post Filter Beta"),
            DfControl::MinThreshDb => write!(f, "Min processing threshold (dB)"),
            DfControl::MaxErbThreshDb => write!(f, "Max ERB processing threshold (dB)"),
            DfControl::MaxDfThreshDb => write!(f, "Max DF processing threshold (dB)"),
            DfControl::MinBufferFrames => write!(f, "Min Processing Buffer (frames)"),
        }
    }
}

struct DfControlHistory {
    atten_lim: f32,
    pf_beta: f32,
    min_thresh_db: f32,
    max_erb_thresh_db: f32,
    max_df_thresh_db: f32,
    min_buffer_frames: f32,
}
impl Default for DfControlHistory {
    fn default() -> Self {
        Self {
            atten_lim: 100.,
            pf_beta: 0.0,
            min_thresh_db: -10.,
            max_erb_thresh_db: 30.,
            max_df_thresh_db: 20.,
            min_buffer_frames: 0.,
        }
    }
}
impl DfControlHistory {
    fn get(&self, c: &DfControl) -> f32 {
        match c {
            DfControl::AttenLim => self.atten_lim,
            DfControl::PfBeta => self.pf_beta,
            DfControl::MinThreshDb => self.min_thresh_db,
            DfControl::MaxErbThreshDb => self.max_erb_thresh_db,
            DfControl::MaxDfThreshDb => self.max_df_thresh_db,
            DfControl::MinBufferFrames => self.min_buffer_frames,
        }
    }
    fn set(&mut self, c: &DfControl, v: f32) {
        match c {
            DfControl::AttenLim => self.atten_lim = v,
            DfControl::PfBeta => self.pf_beta = v,
            DfControl::MinThreshDb => self.min_thresh_db = v,
            DfControl::MaxErbThreshDb => self.max_erb_thresh_db = v,
            DfControl::MaxDfThreshDb => self.max_df_thresh_db = v,
            DfControl::MinBufferFrames => self.min_buffer_frames = v,
        }
    }
}

impl DfPlugin {
    fn worker_finished(&self) -> bool {
        self.worker.as_ref().is_none_or(JoinHandle::is_finished)
    }

    fn fail_worker(&mut self, reason: &str) {
        if !self.worker_failed {
            log::error!("DF {} | {reason}; bypassing processing", self.id);
            self.worker_failed = true;
        }
        self.cancelled.store(true, Ordering::Release);
    }

    #[cfg(feature = "dbus")]
    fn stop_dbus(&mut self) {
        if let Some((handle, done, cancelled)) = self.dbus.take() {
            cancelled.store(true, Ordering::Release);
            done.notify(1);
            match handle.join() {
                Ok(_) => log::debug!("{} | dbus thread joined", self.id),
                Err(error) => log::error!("{} | dbus thread error: {:?}", self.id, error),
            }
        }
    }
}

impl Drop for DfPlugin {
    fn drop(&mut self) {
        #[cfg(feature = "dbus")]
        self.stop_dbus();
        self.cancelled.store(true, Ordering::Release);
        if let Some(handle) = self.worker.take() {
            if let Err(error) = handle.join() {
                log::error!("DF {} | Worker thread error: {:?}", self.id, error);
            }
        }
    }
}

impl Plugin for DfPlugin {
    fn activate(&mut self) {
        log::info!("DF {} | activate", self.id);
        #[cfg(feature = "dbus")]
        {
            let done = Arc::new(Event::new());
            let cancelled = Arc::new(AtomicBool::new(false));
            let (worker, ready) = get_dbus_worker(
                self.control_tx.clone(),
                Arc::clone(&done),
                self.id.clone(),
                Arc::clone(&cancelled),
            );
            let handle = thread::spawn(worker);
            match ready.recv_timeout(DBUS_INIT_TIMEOUT) {
                Ok(Ok(())) => {
                    self.dbus = Some((handle, done, cancelled));
                    log::debug!("dbus thread spawned");
                }
                Ok(Err(error)) => {
                    let _ = handle.join();
                    log::error!("Failed to init dbus session: {error}");
                }
                Err(RecvTimeoutError::Disconnected) => {
                    let _ = handle.join();
                    log::error!("dbus thread terminated during initialization");
                }
                Err(RecvTimeoutError::Timeout) => {
                    cancelled.store(true, Ordering::Release);
                    done.notify(1);
                    self.dbus = Some((handle, done, cancelled));
                    log::error!(
                        "dbus initialization timed out after {:.1}s",
                        DBUS_INIT_TIMEOUT.as_secs_f32()
                    );
                }
            }
        }
    }
    fn deactivate(&mut self) {
        log::info!("DF {} | deactivate", self.id);
        #[cfg(feature = "dbus")]
        self.stop_dbus();
    }
    fn run<'a>(&mut self, sample_count: usize, ports: &[&'a PortConnection<'a>]) {
        let t0 = Instant::now();

        let mut i = 0;
        let mut inputs = Vec::with_capacity(self.ch);
        let mut outputs = Vec::with_capacity(self.ch);
        for _ in 0..self.ch {
            inputs.push(ports[i].unwrap_audio());
            i += 1;
        }
        for _ in 0..self.ch {
            outputs.push(ports[i].unwrap_audio_mut());
            i += 1;
        }
        if self.worker_failed || self.worker_finished() {
            if !self.worker_failed {
                self.fail_worker("Worker terminated");
            }
            for (i_ch, o_ch) in inputs.iter().zip(outputs.iter_mut()) {
                o_ch.copy_from_slice(i_ch);
            }
            return;
        }
        for p in ports[i..].iter() {
            let &v = p.unwrap_control();
            let c = DfControl::from_port_name(p.port.name);
            if c == DfControl::AttenLim && v >= 100. {
                for (i_ch, o_ch) in inputs.iter().zip(outputs.iter_mut()) {
                    for (&i, o) in i_ch.iter().zip(o_ch.iter_mut()) {
                        *o = i
                    }
                }
            }
            if v != self.control_hist.get(&c) {
                match self.control_tx.try_send((c, v)) {
                    Ok(()) => self.control_hist.set(&c, v),
                    Err(TrySendError::Full(_)) => {
                        log::debug!("DF {} | Worker control queue full", self.id)
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        self.fail_worker("Worker control channel disconnected");
                    }
                }
            }
        }

        if self.worker_failed {
            for (i_ch, o_ch) in inputs.iter().zip(outputs.iter_mut()) {
                o_ch.copy_from_slice(i_ch);
            }
            return;
        }

        {
            let i_q = &mut self.i_tx.lock().unwrap();
            for (i_ch, i_q_ch) in inputs.iter().zip(i_q.iter_mut()) {
                for &i in i_ch.iter() {
                    i_q_ch.push_back(i)
                }
            }
        }

        'outer: loop {
            {
                let o_q = &mut self.o_rx.lock().unwrap();
                if o_q[0].len() >= sample_count {
                    for (o_q_ch, o_ch) in o_q.iter_mut().zip(outputs.iter_mut()) {
                        for o in o_ch.iter_mut() {
                            *o = o_q_ch.pop_front().unwrap();
                        }
                    }
                    break 'outer;
                }
            }
            if self.worker_finished() {
                self.fail_worker("Worker terminated");
                for (i_ch, o_ch) in inputs.iter().zip(outputs.iter_mut()) {
                    o_ch.copy_from_slice(i_ch);
                }
                return;
            }
            if t0.elapsed() >= self.sleep_duration * (WORKER_STALL_FRAMES * 5) {
                self.fail_worker("Worker missed the processing deadline");
                for (i_ch, o_ch) in inputs.iter().zip(outputs.iter_mut()) {
                    o_ch.copy_from_slice(i_ch);
                }
                return;
            }
            sleep(self.sleep_duration);
        }

        let td = t0.elapsed();
        let t_audio = sample_count as f32 / self.sr as f32;
        let rtf = td.as_secs_f32() / t_audio;
        if rtf >= 1. {
            log::warn!(
                "DF {} | Underrun detected (RTF: {:.2}). Processing too slow!",
                self.id,
                rtf
            );
            if self.proc_delay >= self.sr {
                panic!(
                    "DF {} | Processing too slow! Please upgrade your CPU. Try to decrease 'Max DF processing threshold (dB)'.",
                    self.id,
                );
            }
            self.proc_delay += self.frame_size;
            self.t_proc_change = 0;
            log::info!(
                "DF {} | Increasing processing latency to {:.1}ms",
                self.id,
                self.proc_delay as f32 * 1000. / self.sr as f32
            );
            for o_ch in self.o_rx.lock().unwrap().iter_mut() {
                for _ in 0..self.frame_size {
                    o_ch.push_back(0f32)
                }
            }
        } else if self.t_proc_change > 10 * self.sr / self.frame_size
            && rtf < 0.5
            && self.proc_delay
                >= self.frame_size * (1 + self.control_hist.min_buffer_frames as usize)
        {
            // Reduce delay again
            let dropped_samples = {
                let o_q = &mut self.o_rx.lock().unwrap();
                if o_q[0].len() < self.frame_size {
                    false
                } else {
                    for o_q_ch in o_q.iter_mut().take(self.frame_size) {
                        o_q_ch.pop_front().unwrap();
                    }
                    true
                }
            };
            if dropped_samples {
                self.proc_delay -= self.frame_size;
                self.t_proc_change = 0;
                log::info!(
                    "DF {} | Decreasing processing latency to {:.1}ms",
                    self.id,
                    self.proc_delay as f32 * 1000. / self.sr as f32
                );
            }
        }
        self.t_proc_change += 1;
    }
}

#[cfg(feature = "dbus")]
fn build_dbus_session<I>(control: I) -> Result<zbus::blocking::Connection, zbus::Error>
where
    I: Interface,
{
    ConnectionBuilder::session()?
        .name(DBUS_NAME)?
        .serve_at(DBUS_PATH, control)?
        .build()
}
#[cfg(feature = "dbus")]
fn get_dbus_worker(
    tx: ControlProd,
    done: Arc<Event>,
    id: String,
    cancelled: Arc<AtomicBool>,
) -> (impl FnOnce(), Receiver<Result<(), String>>) {
    let (ready_tx, ready_rx) = sync_channel(1);
    let worker = move || {
        log::debug!("{id} | Initializing dbus server");
        let done_listener = done.clone().listen();
        let control = DfDbusControl { tx: tx.clone() };
        let con = match build_dbus_session(control) {
            Ok(connection) => connection,
            Err(error) => {
                let _ = ready_tx.send(Err(error.to_string()));
                return;
            }
        };
        if cancelled.load(Ordering::Acquire) {
            let _ = con.release_name(DBUS_NAME);
            return;
        }
        if ready_tx.send(Ok(())).is_err() {
            let _ = con.release_name(DBUS_NAME);
            return;
        }
        done_listener.wait();
        if let Err(error) = con.release_name(DBUS_NAME) {
            log::error!("{id} | Failed to release dbus name: {error}");
        }
        log::debug!("{id} | Got done notification. Releasing dbus name");
    };
    (worker, ready_rx)
}

#[cfg(feature = "dbus")]
struct DfDbusControl {
    tx: ControlProd,
}

#[cfg(feature = "dbus")]
impl DfDbusControl {
    fn send(&self, control: DfControl, value: f32) -> zbus::fdo::Result<()> {
        self.tx
            .try_send((control, value))
            .map_err(|error| zbus::fdo::Error::Failed(error.to_string()))
    }
}

#[cfg(feature = "dbus")]
#[interface(name = "org.deepfilter.DeepFilterLadspa")]
impl DfDbusControl {
    fn atten_lim(&self, lim: u32) -> zbus::fdo::Result<()> {
        self.send(DfControl::AttenLim, lim as f32)
    }
    fn pf_beta(&self, beta: f32) -> zbus::fdo::Result<()> {
        self.send(DfControl::PfBeta, beta)
    }
    fn min_processing_thresh(&self, thresh: i32) -> zbus::fdo::Result<()> {
        self.send(DfControl::MinThreshDb, thresh as f32)
    }
    fn max_erb_thresh(&self, thresh: i32) -> zbus::fdo::Result<()> {
        self.send(DfControl::MaxErbThreshDb, thresh as f32)
    }
    fn max_df_thresh(&self, thresh: i32) -> zbus::fdo::Result<()> {
        self.send(DfControl::MaxDfThreshDb, thresh as f32)
    }
}

#[no_mangle]
pub fn get_ladspa_descriptor(index: u64) -> Option<PluginDescriptor> {
    match index {
        0 => Some(PluginDescriptor {
            unique_id: ID_MONO,
            label: "deep_filter_mono",
            properties: ladspa::PROP_NONE,
            name: "DeepFilter Mono",
            maker: "Hendrik Schröter",
            copyright: "MIT/Apache",
            ports: vec![
                Port {
                    name: "Audio In",
                    desc: PortDescriptor::AudioInput,
                    ..Default::default()
                },
                Port {
                    name: "Audio Out",
                    desc: PortDescriptor::AudioOutput,
                    ..Default::default()
                },
                Port {
                    name: "Attenuation Limit (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(ATTEN_LIM_DEF),
                    lower_bound: Some(ATTEN_LIM_MIN),
                    upper_bound: Some(ATTEN_LIM_MAX),
                },
                Port {
                    name: "Min processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(MIN_PROC_THRESH_DEF),
                    lower_bound: Some(MIN_PROC_THRESH_MIN),
                    upper_bound: Some(MIN_PROC_THRESH_MAX),
                },
                Port {
                    name: "Max ERB processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(MAX_ERB_BUF_DEF),
                    lower_bound: Some(MAX_ERB_BUF_MIN),
                    upper_bound: Some(MAX_ERB_BUF_MAX),
                },
                Port {
                    name: "Max DF processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(MAX_DF_BUF_DEF),
                    lower_bound: Some(MAX_DF_BUF_MIN),
                    upper_bound: Some(MAX_DF_BUF_MAX),
                },
                Port {
                    name: "Min Processing Buffer (frames)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(MIN_PROC_BUF_DEF),
                    lower_bound: Some(MIN_PROC_BUF_MIN),
                    upper_bound: Some(MIN_PROC_BUF_MAX),
                },
                Port {
                    name: "Post Filter Beta",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(PF_BETA_DEF),
                    lower_bound: Some(PF_BETA_MIN),
                    upper_bound: Some(PF_BETA_MAX),
                },
            ],
            new: |d, sr| Box::new(get_new_df(1)(d, sr)),
        }),
        1 => Some(PluginDescriptor {
            unique_id: ID_STEREO,
            label: "deep_filter_stereo",
            properties: ladspa::PROP_NONE,
            name: "DeepFilter Stereo",
            maker: "Hendrik Schröter",
            copyright: "MIT/Apache",
            ports: vec![
                Port {
                    name: "Audio In L",
                    desc: PortDescriptor::AudioInput,
                    ..Default::default()
                },
                Port {
                    name: "Audio In R",
                    desc: PortDescriptor::AudioInput,
                    ..Default::default()
                },
                Port {
                    name: "Audio Out L",
                    desc: PortDescriptor::AudioOutput,
                    ..Default::default()
                },
                Port {
                    name: "Audio Out R",
                    desc: PortDescriptor::AudioOutput,
                    ..Default::default()
                },
                Port {
                    name: "Attenuation Limit (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(ATTEN_LIM_DEF),
                    lower_bound: Some(ATTEN_LIM_MIN),
                    upper_bound: Some(ATTEN_LIM_MAX),
                },
                Port {
                    name: "Min processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(MIN_PROC_THRESH_DEF),
                    lower_bound: Some(MIN_PROC_THRESH_MIN),
                    upper_bound: Some(MIN_PROC_THRESH_MAX),
                },
                Port {
                    name: "Max ERB processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(MAX_ERB_BUF_DEF),
                    lower_bound: Some(MAX_ERB_BUF_MIN),
                    upper_bound: Some(MAX_ERB_BUF_MAX),
                },
                Port {
                    name: "Max DF processing threshold (dB)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(MAX_DF_BUF_DEF),
                    lower_bound: Some(MAX_DF_BUF_MIN),
                    upper_bound: Some(MAX_DF_BUF_MAX),
                },
                Port {
                    name: "Min Processing Buffer (frames)",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(MIN_PROC_BUF_DEF),
                    lower_bound: Some(MIN_PROC_BUF_MIN),
                    upper_bound: Some(MIN_PROC_BUF_MAX),
                },
                Port {
                    name: "Post Filter Beta",
                    desc: PortDescriptor::ControlInput,
                    hint: None,
                    default: Some(PF_BETA_DEF),
                    lower_bound: Some(PF_BETA_MIN),
                    upper_bound: Some(PF_BETA_MAX),
                },
            ],
            new: |d, sr| Box::new(get_new_df(2)(d, sr)),
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_readiness_reports_success() {
        let (tx, rx) = sync_channel(1);
        tx.send(Ok(())).unwrap();

        assert_eq!(wait_for_worker_ready(&rx, Duration::from_secs(1)), Ok(()));
    }

    #[test]
    fn worker_readiness_reports_initialization_error() {
        let (tx, rx) = sync_channel(1);
        tx.send(Err("invalid model".to_string())).unwrap();

        assert_eq!(
            wait_for_worker_ready(&rx, Duration::from_secs(1)),
            Err("invalid model".to_string())
        );
    }

    #[test]
    fn worker_readiness_reports_early_termination() {
        let (tx, rx) = sync_channel::<Result<(), String>>(1);
        drop(tx);

        let error = wait_for_worker_ready(&rx, Duration::from_secs(1)).unwrap_err();
        assert!(error.contains("worker terminated before initialization completed"));
    }

    #[test]
    fn worker_readiness_times_out() {
        let (_tx, rx) = sync_channel::<Result<(), String>>(1);

        let error = wait_for_worker_ready(&rx, Duration::from_millis(1)).unwrap_err();
        assert!(error.contains("initialization timed out"));
    }
}
