use std::{
    fs::File,
    io::{BufWriter, Write},
    time::Duration,
};

use anyhow::Context;
use shine_rs::{Mp3Encoder, Mp3EncoderConfig, StereoMode};

pub const SAMPLE_RATE: u32 = 24_000;
pub const CHANNELS: u8 = 1;

/// Convert normalized [-1, 1] float to i16 PCM.
fn f32_to_i16(x: f32) -> i16 {
    let x = x.clamp(-1.0, 1.0);
    (x * i16::MAX as f32) as i16
}

/// Duration -> number of PCM frames (per-channel samples) at sample_rate.
/// Uses floor; errors if duration is too small for at least 1 frame.
fn frames_for_duration(sample_rate: u32, d: Duration) -> anyhow::Result<u64> {
    let ns: u128 = d.as_nanos();
    let frames: u128 = (ns * sample_rate as u128) / 1_000_000_000u128;
    anyhow::ensure!(frames > 0, "segment duration too small");
    anyhow::ensure!(frames <= u64::MAX as u128, "segment duration too large");
    Ok(frames as u64)
}

pub struct Mp3Splitter {
    prefix: String,
    index: u32,

    config: Mp3EncoderConfig,

    /// Max frames (per channel) per file.
    frames_per_file: u64,
    written_frames: u64,

    out: Option<BufWriter<File>>,
    enc: Option<Mp3Encoder>,

    /// Scratch buffer for PCM conversion (interleaved i16).
    pcm_i16: Vec<i16>,
}

impl Mp3Splitter {
    /// `segment_duration`: target max audio duration per MP3 file.
    /// `config`: MP3 encoder settings (sample_rate/bitrate/channels/stereo_mode).
    pub fn new(
        prefix: impl Into<String>,
        config: Mp3EncoderConfig,
        segment_duration: Duration,
    ) -> anyhow::Result<Self> {
        // Validate early so we fail before writing any files.
        config.validate().context("invalid MP3 encoder config")?;

        let frames_per_file = frames_for_duration(config.sample_rate, segment_duration)?;
        tracing::info!(
            "MP3 split duration {:?} => {} frames per file (sr={}, ch={})",
            segment_duration,
            frames_per_file,
            config.sample_rate,
            config.channels
        );

        Ok(Self {
            prefix: prefix.into(),
            index: 0,
            config,
            frames_per_file,
            written_frames: 0,
            out: None,
            enc: None,
            pcm_i16: Vec::new(),
        })
    }

    fn finish_current(&mut self) -> anyhow::Result<()> {
        // If nothing opened yet, nothing to do.
        if self.out.is_none() && self.enc.is_none() {
            return Ok(());
        }

        // Take writer first so we can write tail bytes then drop/flush.
        let mut out = self
            .out
            .take()
            .context("internal error: encoder exists without writer")?;

        if let Some(mut enc) = self.enc.take() {
            // Finish pads the last partial MP3 frame (if any) and flushes. (Normal MP3 behavior.)
            let tail = enc.finish().context("mp3 encoder finish failed")?;
            if !tail.is_empty() {
                out.write_all(&tail).context("failed writing mp3 tail")?;
            }
        }

        out.flush().context("failed flushing mp3 output")?;
        self.written_frames = 0;
        Ok(())
    }

    fn open_next(&mut self) -> anyhow::Result<()> {
        self.finish_current()?;

        let path = format!("{}_{:03}.mp3", self.prefix, self.index);
        self.index += 1;

        let file = File::create(&path).with_context(|| format!("create {}", path))?;
        let out = BufWriter::new(file);

        let enc = Mp3Encoder::new(self.config.clone()).context("create mp3 encoder")?;

        self.out = Some(out);
        self.enc = Some(enc);
        self.written_frames = 0;

        Ok(())
    }

    /// Write interleaved f32 samples (`[L, R, L, R, ...]` for stereo; `[M, M, ...]` for mono),
    /// splitting to new MP3 files once `segment_duration` worth of frames is reached.
    pub fn write_f32_interleaved(&mut self, samples: &[f32]) -> anyhow::Result<()> {
        let ch = self.config.channels as usize;
        anyhow::ensure!(ch == 1 || ch == 2, "only 1 or 2 channels supported");
        anyhow::ensure!(
            samples.len().is_multiple_of(ch),
            "interleaved buffer length must be a multiple of channels"
        );

        if self.enc.is_none() {
            self.open_next()?;
        }

        let total_frames = samples.len() / ch;
        let mut frame_offset = 0usize;

        while frame_offset < total_frames {
            let remaining_frames_in_file = (self.frames_per_file - self.written_frames) as usize;
            if remaining_frames_in_file == 0 {
                // Close current file & start next segment
                self.open_next()?;
                continue;
            }

            let take_frames = remaining_frames_in_file.min(total_frames - frame_offset);

            let start = frame_offset * ch;
            let end = (frame_offset + take_frames) * ch;

            self.pcm_i16.clear();
            self.pcm_i16.reserve(end - start);
            for &s in &samples[start..end] {
                self.pcm_i16.push(f32_to_i16(s));
            }

            let enc = self.enc.as_mut().unwrap();
            let mp3_blocks = enc
                .encode_interleaved(&self.pcm_i16)
                .context("mp3 encode_interleaved failed")?;

            let out = self.out.as_mut().unwrap();
            for b in mp3_blocks {
                out.write_all(&b)
                    .context("failed writing mp3 frame block")?;
            }

            self.written_frames += take_frames as u64;
            frame_offset += take_frames;

            // If we exactly filled the segment AND still have more input to write in this call,
            // rotate immediately (so one input call can produce multiple files).
            if self.written_frames == self.frames_per_file && frame_offset < total_frames {
                self.open_next()?;
            }
        }

        Ok(())
    }

    /// Convenience for mono, like your original API.
    pub fn write_f32_mono(&mut self, samples: &[f32]) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.config.channels == 1,
            "config.channels must be 1 for mono"
        );
        self.write_f32_interleaved(samples)
    }

    pub fn finalize(mut self) -> anyhow::Result<()> {
        self.finish_current()
    }
}

/// Example config matching your constants (24kHz mono).
pub fn default_mono_24k_config(bitrate_kbps: u32) -> Mp3EncoderConfig {
    Mp3EncoderConfig::new()
        .sample_rate(SAMPLE_RATE)
        .bitrate(bitrate_kbps)
        .channels(CHANNELS)
        .stereo_mode(StereoMode::Mono)
}
