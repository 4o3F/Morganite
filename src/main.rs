use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
    sync::Arc,
    time::Duration,
};

use chrono::Local;
use clap::Parser;
use hound::{SampleFormat, WavSpec};
use kokoro_tts::Voice;
use tokio::{
    sync::{Semaphore, mpsc},
    task::JoinSet,
};
use tracing::Level;
use tracing_appender::non_blocking;
use tracing_indicatif::{IndicatifLayer, span_ext::IndicatifSpanExt, style::ProgressStyle};
use tracing_subscriber::{fmt, layer::SubscriberExt};
use tracing_unwrap::ResultExt;

mod tts;
mod utils;
mod writer;

#[derive(clap::Parser)]
struct Cli {
    /// Path for the file to be converted
    text_file: String,

    /// Path for onnx tts model
    #[arg(long, short, default_value = "kokoro-v1.1-zh.onnx")]
    tts_model: String,

    /// Path for voice bin model
    #[arg(long, short, default_value = "voices-v1.1-zh.bin")]
    voice_model: String,

    /// Voice name, e.g. zf_048, zm_029, af_maple
    #[arg(long, value_parser = utils::parse_voice, default_value = "zf_048")]
    voice: Voice,

    /// Speech speed
    #[arg(long, default_value_t = 1.0)]
    speed: f32,

    /// Concurrency
    #[arg(long, default_value_t = 4)]
    concurrency: usize,
}

type Msg = (usize, anyhow::Result<(Vec<f32>, Duration)>);

#[tokio::main]
async fn main() {
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let target_dir = PathBuf::from(&timestamp);
    if !target_dir.exists() {
        std::fs::create_dir(&target_dir).expect("Failed to create target dir");
    }
    let file_path = format!("{}/app.log", timestamp);

    let file_appender = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .append(false)
        .write(true)
        .open(&file_path)
        .expect("failed to create log file");
    let (non_blocking_writer, _guard) = non_blocking(file_appender);

    let indicatif_layer = IndicatifLayer::new();

    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_level(true)
        .finish()
        .with(indicatif_layer)
        .with(fmt::Layer::default().with_writer(non_blocking_writer));

    tracing::subscriber::set_global_default(subscriber).expect_or_log("Init tracing failed");

    let cli = Cli::parse();

    if !PathBuf::from(&cli.tts_model).exists() {
        tracing::error!("Unable to finx ONNX TTS model file {}", cli.tts_model);
        return;
    }

    if !PathBuf::from(&cli.voice_model).exists() {
        tracing::error!("Unable to find voice model file {}", cli.voice_model);
        return;
    }

    tracing::info!("Using ONNX TTS model {}", cli.tts_model);
    tracing::info!("Using voice model {}", cli.voice_model);

    if !PathBuf::from(&cli.text_file).exists() {
        tracing::error!("Unable to find text file {}", cli.text_file);
        return;
    }

    let tts_engine = tts::init_tts(cli.tts_model, cli.voice_model, cli.concurrency).await;

    tracing::info!("Initialized KokoroTTS engine");

    let spec = WavSpec {
        channels: writer::CHANNELS,
        sample_rate: writer::SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut wav = writer::WavSplitter::new(format!("{}/audio", timestamp), spec)
        .expect_or_log("Failed to create WAV split writer");

    // let text_file = PathBuf::from(cli.text_file);
    let text_file = File::open(&cli.text_file).expect_or_log("Failed to open text file");
    let total_lines = {
        let text_reader = BufReader::new(&text_file);
        text_reader
            .lines()
            .map(|l| {
                l.expect_or_log("Failed to get line of text file")
                    .trim()
                    .to_string()
            })
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
    };
    tracing::info!("Target file total {} line", total_lines.len());

    let voice = utils::change_voice_speed(cli.voice, cli.speed);

    let sem = Arc::new(Semaphore::new(cli.concurrency * 2));
    let (tx, mut rx) = mpsc::channel::<Msg>(cli.concurrency * 2);

    let producer: tokio::task::JoinHandle<anyhow::Result<()>> = tokio::spawn(async move {
        let mut set = JoinSet::<anyhow::Result<()>>::new();

        let header_span = tracing::info_span!("task");
        header_span.pb_set_style(
            &ProgressStyle::with_template("{spinner} {msg}\n{wide_bar} {pos}/{len}").unwrap(),
        );
        header_span.pb_set_length(total_lines.len() as u64);
        header_span.pb_set_message("Processing items");
        header_span.pb_set_finish_message("All items processed");

        let header_span_enter = header_span.enter();

        for (line_index, line) in total_lines.iter().enumerate() {
            let line = line.clone();
            if line.is_empty() {
                unreachable!()
            }

            let permit = sem.clone().acquire_owned().await?;

            let tx2 = tx.clone();
            let header_span = header_span.clone();
            let current_audio_idx = line_index;

            set.spawn(async move {
                let _permit = permit;
                tracing::info!("Audio idx {} started", current_audio_idx);

                let res = tts_engine
                    .synth::<String>(line, voice)
                    .await
                    .map_err(|e| anyhow::anyhow!("{}", e));

                tracing::info!("Audio idx {} finished", current_audio_idx);
                let _ = tx2.send((current_audio_idx, res)).await;
                tracing::info!("Audio idx {} sent to channel", current_audio_idx);

                header_span.pb_inc(1);
                Ok(())
            });
        }
        drop(tx);
        while let Some(r) = set.join_next().await {
            r??;
        }
        drop(header_span_enter);
        Ok(())
    });

    let mut next_expected: usize = 0;
    let mut buffer: BTreeMap<usize, (Vec<f32>, Duration)> = BTreeMap::new();

    while let Some((idx, res)) = rx.recv().await {
        let (audio, took) = res.expect_or_log("Failed to get synth result");
        buffer.insert(idx, (audio, took));

        while let Some((audio, took)) = buffer.remove(&next_expected) {
            wav.write_f32_mono(&audio)
                .expect_or_log("Failed to write to wav");
            tracing::info!("Audio idx {next_expected} took {:?}", took);
            next_expected += 1;
        }
    }

    producer
        .await
        .unwrap()
        .expect_or_log("Failed to finish synth task");

    wav.finalize().expect_or_log("Failed to finalize wav write");
}
