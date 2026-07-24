use std::env;
use std::path::PathBuf;
use std::process::exit;

use clap::{Parser, ValueHint};
use crossbeam_channel::unbounded;
use iced::widget::{self, column, container, image, row, slider, text, Container, Image};
use iced::{alignment, Alignment, ContentFit, Element, Length, Settings, Subscription, Task};
use image_rs::{imageops, Rgba, RgbaImage};

mod capture;
mod cmap;
use capture::*;

/// Simple program to sample from a hd5 dataset directory
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to model tar.gz
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    model: Option<PathBuf>,
    /// Logging verbosity
    #[arg(
        long,
        short = 'v',
        action = clap::ArgAction::Count,
        global = true,
        help = "Increase logging verbosity with multiple `-vv`",
    )]
    verbose: u8,
}

pub fn main() -> iced::Result {
    let args = Args::parse();
    let level = match args.verbose {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    let tract_level = match args.verbose {
        0..=3 => log::LevelFilter::Error,
        4 => log::LevelFilter::Info,
        5 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    if args.model.is_some() {
        *MODEL_PATH.lock().unwrap() = args.model;
    }

    capture::INIT_LOGGER.call_once(|| {
        env_logger::Builder::from_env(env_logger::Env::default())
            .filter_level(level)
            .filter_module("tract_onnx", tract_level)
            .filter_module("tract_hir", tract_level)
            .filter_module("tract_core", tract_level)
            .filter_module("tract_linalg", tract_level)
            .filter_module("iced_winit", log::LevelFilter::Error)
            .filter_module("iced_wgpu", log::LevelFilter::Error)
            .filter_module("wgpu_core", log::LevelFilter::Error)
            .filter_module("wgpu_hal", log::LevelFilter::Error)
            .filter_module("naga", log::LevelFilter::Error)
            .filter_module("crossfont", log::LevelFilter::Error)
            .filter_module("cosmic_text", log::LevelFilter::Error)
            .format(capture::log_format)
            .init();
    });

    iced::application(SpecView::new, SpecView::update, SpecView::view)
        .title(SpecView::title)
        .subscription(SpecView::subscription)
        .settings(Settings::default())
        .run()
}

struct SpecView {
    df_worker: DeepFilterCapture,
    lsnr: f32,
    atten_lim: f32,
    post_filter_beta: f32,
    min_threshdb: f32,
    max_erbthreshdb: f32,
    max_dfthreshdb: f32,
    noisy_spec: SpecImage,
    enh_spec: SpecImage,
    noisy_img: image::Handle,
    enh_img: image::Handle,
    r_lsnr: RecvLsnr,
    r_noisy: RecvSpec,
    r_enh: RecvSpec,
    s_controls: SendControl,
}

#[derive(Debug, Clone, Copy)]
pub enum Message {
    Tick,
    AttenLimChanged(f32),
    PostFilterChanged(f32),
    MinThreshDbChanged(f32),
    MaxErbThreshDbChanged(f32),
    MaxDfThreshDbChanged(f32),
    Exit,
}

struct SpecImage {
    im: RgbaImage,
    n_frames: u32,
    n_freqs: u32,
    vmin: f32,
    vmax: f32,
}

impl SpecImage {
    fn new(n_frames: u32, n_freqs: u32, vmin: f32, vmax: f32) -> Self {
        Self {
            // Store image transposed so we can iterate over rows quickly
            im: RgbaImage::new(n_freqs, n_frames),
            n_frames,
            n_freqs,
            vmin,
            vmax,
        }
    }
    fn w(&self) -> usize {
        self.n_frames as usize
    }
    fn h(&self) -> usize {
        self.n_freqs as usize
    }
    fn update<I>(&mut self, specs: I, mut n_specs: usize)
    where
        I: Iterator<Item = Box<[f32]>>,
    {
        if n_specs == 0 {
            return;
        }
        if n_specs >= self.n_frames as usize {
            // Just drop a few
            n_specs = self.n_frames as usize - 1;
        }
        for (spec, im_row) in specs.take(n_specs).zip(self.im.rows_mut()) {
            for (s, x) in spec.iter().zip(im_row) {
                // clamp and normalize
                let v = (s.min(self.vmax).max(self.vmin) - self.vmin) / (self.vmax - self.vmin);
                *x = Rgba(cmap::CMAP_INFERNO[(v * 255.) as usize]);
            }
        }
        let (w, h) = (self.w(), self.h());
        self.im.rotate_left((w - n_specs) * 4 * h);
    }
    fn image_handle(&self) -> image::Handle {
        let imt_buf = imageops::rotate270(&self.im).as_raw().to_vec();
        image::Handle::from_rgba(self.n_frames, self.n_freqs, imt_buf)
    }
}

impl SpecView {
    fn new() -> (Self, Task<Message>) {
        let (s_lsnr, r_lsnr) = unbounded();
        let (s_noisy, r_noisy) = unbounded();
        let (s_enh, r_enh) = unbounded();
        let (s_controls, r_controls) = unbounded();

        let model_path = env::var("DF_MODEL").ok().map(PathBuf::from);
        let df_worker = DeepFilterCapture::new(
            model_path,
            Some(s_lsnr),
            Some(s_noisy),
            Some(s_enh),
            Some(r_controls),
        )
        .expect("Failed to initialize DeepFilterNet audio capturing");

        let w = (df_worker.sr / df_worker.frame_size * 10) as u32;
        let freq_res = df_worker.sr / 2 / (df_worker.freq_size - 1);
        let h = (8000 / freq_res) as u32;
        let noisy_spec = SpecImage::new(w, h, -100., -10.);
        let enh_spec = SpecImage::new(w, h, -100., -10.);
        let noisy_img = noisy_spec.image_handle();
        let enh_img = enh_spec.image_handle();
        (
            Self {
                df_worker,
                lsnr: 0.,
                atten_lim: 100.,
                post_filter_beta: 0.,
                min_threshdb: -15.,
                max_erbthreshdb: 35.,
                max_dfthreshdb: 35.,
                noisy_spec,
                enh_spec,
                r_lsnr,
                r_noisy,
                r_enh,
                s_controls,
                noisy_img,
                enh_img,
            },
            Task::none(),
        )
    }

    fn title(&self) -> String {
        "DeepFilterNet Demo".to_string()
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Exit => {
                self.df_worker.should_stop().expect("Failed to stop DF worker");
                exit(0);
            }
            Message::Tick => {
                self.update_lsnr();
                self.update_noisy();
                self.update_enh();
            }
            Message::AttenLimChanged(v) => {
                self.atten_lim = v;
                self.s_controls
                    .send((DfControl::AttenLim, v))
                    .expect("Failed to send DfControl")
            }
            Message::PostFilterChanged(v) => {
                self.post_filter_beta = v;
                self.s_controls
                    .send((DfControl::PostFilterBeta, v))
                    .expect("Failed to send DfControl")
            }
            Message::MinThreshDbChanged(v) => {
                self.min_threshdb = v;
                self.s_controls
                    .send((DfControl::MinThreshDb, v))
                    .expect("Failed to send DfControl")
            }
            Message::MaxErbThreshDbChanged(v) => {
                self.max_erbthreshdb = v;
                self.s_controls
                    .send((DfControl::MaxErbThreshDb, v))
                    .expect("Failed to send DfControl")
            }
            Message::MaxDfThreshDbChanged(v) => {
                self.max_dfthreshdb = v;
                self.s_controls
                    .send((DfControl::MaxDfThreshDb, v))
                    .expect("Failed to send DfControl")
            }
        }
        Task::none()
    }

    fn view(&self) -> Element<'_, Message> {
        let content = column![row![
            text("DeepFilterNet Demo").size(40).width(Length::Fill),
            button("exit").on_press(Message::Exit)
        ]
        .width(1000),];
        #[cfg(feature = "thresholds")]
        let content = {
            content
                .push(slider_view(
                    "Threshold Min [dB]",
                    self.min_threshdb,
                    -15.,
                    35.,
                    Message::MinThreshDbChanged,
                    1000,
                    0,
                    3.,
                ))
                .push(slider_view(
                    "Threshold ERB Max [dB]",
                    self.max_erbthreshdb,
                    -15.,
                    35.,
                    Message::MaxErbThreshDbChanged,
                    1000,
                    0,
                    3.,
                ))
                .push(slider_view(
                    "Threshold DF  Max [dB]",
                    self.max_dfthreshdb,
                    -15.,
                    35.,
                    Message::MaxDfThreshDbChanged,
                    1000,
                    0,
                    3.,
                ))
        };
        let content = content
            .push(slider_view(
                "Noise Attenuation [dB]",
                self.atten_lim,
                0.,
                100.,
                Message::AttenLimChanged,
                1000,
                0,
                3.,
            ))
            .push(slider_view(
                "Post Filter Beta",
                self.post_filter_beta,
                0.,
                1.,
                Message::PostFilterChanged,
                1000,
                3,
                0.001,
            ))
            .push(self.specs())
            .push(
                row![
                    text("Current SNR:").size(18),
                    text(format!("{:>5.1} dB", self.lsnr))
                        .size(18)
                        .width(80)
                        .align_x(alignment::Horizontal::Right)
                ]
                .spacing(20)
                .align_y(Alignment::End),
            );

        container(content)
            .padding(50)
            .width(Length::Fill)
            .height(Length::Fill)
            .align_x(alignment::Horizontal::Center)
            .align_y(alignment::Vertical::Center)
            .into()
    }

    fn subscription(&self) -> Subscription<Message> {
        iced::time::every(std::time::Duration::from_millis(20)).map(|_| Message::Tick)
    }
}

impl SpecView {
    fn update_lsnr(&mut self) {
        let mut lsnr = 0.;
        let mut n = 0;
        for value in self.r_lsnr.try_iter() {
            lsnr += value;
            n += 1;
        }
        if n > 0 {
            self.lsnr = lsnr / n as f32;
        }
    }

    fn update_noisy(&mut self) {
        let n = self.r_noisy.len();
        if n > 0 {
            self.noisy_spec.update(self.r_noisy.try_iter().take(n), n);
            self.noisy_img = self.noisy_spec.image_handle();
        }
    }

    fn update_enh(&mut self) {
        let n = self.r_enh.len();
        if n > 0 {
            self.enh_spec.update(self.r_enh.try_iter().take(n), n);
            self.enh_img = self.enh_spec.image_handle();
        }
    }
    fn specs(&self) -> Container<'_, Message> {
        container(column![
            spec_view("Noisy", self.noisy_img.clone(), 1000, 250),
            spec_view("DeepFilterNet Enhanced", self.enh_img.clone(), 1000, 250),
        ])
    }
}

fn spec_view(title: &str, im: image::Handle, width: u32, height: u32) -> Element<'_, Message> {
    column![
        text(title).size(24).width(Length::Fill),
        spec_raw(im, width, height)
    ]
    .max_width(width)
    .width(Length::Fill)
    .into()
}
fn spec_raw<'a>(im: image::Handle, width: u32, height: u32) -> Container<'a, Message> {
    container(Image::new(im).width(width).height(height).content_fit(ContentFit::Fill))
        .max_width(width)
        .max_height(height)
        .width(Length::Fill)
        .align_x(alignment::Horizontal::Center)
        .align_y(alignment::Vertical::Center)
}

#[allow(clippy::too_many_arguments)]
fn slider_view<'a>(
    title: &'a str,
    value: f32,
    min: f32,
    max: f32,
    message: impl Fn(f32) -> Message + 'a,
    width: u32,
    precision: usize,
    step: f32,
) -> Element<'a, Message> {
    column![
        text(title).size(18).width(Length::Fill),
        row![
            container(slider(min..=max, value, message).step(step)).width(Length::Fill),
            text(format!("{:.precision$}", value))
                .size(18)
                .width(100)
                .align_x(alignment::Horizontal::Right)
                .align_y(alignment::Vertical::Top),
        ]
    ]
    .max_width(width)
    .width(Length::Fill)
    .into()
}

fn button(text: &str) -> widget::Button<'_, Message> {
    widget::button(text).padding(10)
}
