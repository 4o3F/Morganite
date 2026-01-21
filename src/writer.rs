use std::{fs::File, io::BufWriter};

use hound::{WavSpec, WavWriter};

pub const SAMPLE_RATE: u32 = 24_000;
pub const CHANNELS: u16 = 1;

fn max_samples_per_wav(spec: &WavSpec) -> u64 {
    let bytes_per_sample = (spec.bits_per_sample as u64 / 8) * spec.channels as u64;
    let max_data_bytes = (u32::MAX as u64).saturating_sub(1024); // headroom
    max_data_bytes / bytes_per_sample
}

fn f32_to_i16(x: f32) -> i16 {
    let x = x.clamp(-1.0, 1.0);
    (x * i16::MAX as f32) as i16
}

pub struct WavSplitter {
    prefix: String,
    index: u32,
    spec: WavSpec,
    max_samples: u64,
    written_samples: u64,
    writer: Option<WavWriter<BufWriter<File>>>,
}

impl WavSplitter {
    pub fn new(prefix: impl Into<String>, spec: WavSpec) -> anyhow::Result<Self> {
        let max_samples = max_samples_per_wav(&spec);
        tracing::info!("Single audio file max samples {}", max_samples);
        Ok(Self {
            prefix: prefix.into(),
            index: 0,
            max_samples,
            spec,
            written_samples: 0,
            writer: None,
        })
    }

    fn open_next(&mut self) -> anyhow::Result<()> {
        if let Some(w) = self.writer.take() {
            w.finalize()?;
        }
        let path = format!("{}_{:03}.wav", self.prefix, self.index);
        self.index += 1;

        let file = File::create(path)?;
        let writer = WavWriter::new(BufWriter::new(file), self.spec)?;
        self.writer = Some(writer);
        self.written_samples = 0;
        Ok(())
    }

    pub fn write_f32_mono(&mut self, samples: &[f32]) -> anyhow::Result<()> {
        if self.writer.is_none() {
            self.open_next()?;
        }

        let mut offset = 0usize;
        while offset < samples.len() {
            let remaining = (self.max_samples - self.written_samples) as usize;
            if remaining == 0 {
                self.open_next()?;
                continue;
            }

            let take = remaining.min(samples.len() - offset);
            let chunk = &samples[offset..offset + take];

            let w = self.writer.as_mut().unwrap();
            for &s in chunk {
                w.write_sample(f32_to_i16(s))?;
            }

            self.written_samples += take as u64;
            offset += take;
        }

        tracing::info!("Current written samples {}", self.written_samples);

        Ok(())
    }

    pub fn finalize(mut self) -> anyhow::Result<()> {
        if let Some(w) = self.writer.take() {
            w.finalize()?;
        }
        Ok(())
    }
}
