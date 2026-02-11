use std::{
    collections::BTreeMap,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use anyhow::Context;
use chrono::Local;
use clap::Parser;
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
    /// Path to a single .txt file OR a folder containing multiple .txt files
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

fn is_txt(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("txt"))
        .unwrap_or(false)
}

fn file_stem_string(p: &Path) -> String {
    p.file_stem()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn read_non_empty_lines(path: &Path) -> anyhow::Result<Vec<String>> {
    let f =
        File::open(path).with_context(|| format!("Failed to open text file {}", path.display()))?;
    let reader = BufReader::new(f);
    Ok(reader
        .lines()
        .map(|l| {
            l.expect_or_log("Failed to get line of text file")
                .trim()
                .to_string()
        })
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>())
}

#[tokio::main]
async fn main() {
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();

    // Keep a top-level timestamp folder for logs (and for single-file output, like before)
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

    let input_path = PathBuf::from(&cli.text_file);
    if !input_path.exists() {
        tracing::error!("Unable to find input path {}", cli.text_file);
        return;
    }

    // Build the list of txt files to process
    let (txt_files, folder_mode): (Vec<PathBuf>, bool) = if input_path.is_file() {
        if !is_txt(&input_path) {
            tracing::error!("Input file is not a .txt: {}", input_path.display());
            return;
        }
        (vec![input_path.clone()], false)
    } else if input_path.is_dir() {
        let mut files = std::fs::read_dir(&input_path)
            .expect_or_log("Failed to read input directory")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file() && is_txt(p))
            .collect::<Vec<_>>();
        files.sort();
        if files.is_empty() {
            tracing::error!("No .txt files found in folder {}", input_path.display());
            return;
        }
        (files, true)
    } else {
        tracing::error!(
            "Input path is neither a file nor a directory: {}",
            input_path.display()
        );
        return;
    };

    // Init TTS once; share via Arc so tasks can clone handles safely.
    let tts_engine = Arc::new(tts::init_tts(cli.tts_model, cli.voice_model, cli.concurrency).await);
    tracing::info!("Initialized KokoroTTS engine");

    let voice = utils::change_voice_speed(cli.voice, cli.speed);

    // Process each txt file (single file => one iteration)
    for txt_path in txt_files {
        let file_label = txt_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown.txt")
            .to_string();

        tracing::info!("Processing {}", txt_path.display());

        // Decide mp3 output prefix and ensure output folder exists
        let mp3_prefix = if folder_mode {
            let file_name = file_stem_string(&txt_path);
            let out_dir = PathBuf::from(&timestamp).join(&file_name);
            std::fs::create_dir_all(&out_dir)
                .with_context(|| format!("Failed to create output folder {}", out_dir.display()))
                .unwrap_or_log();
            format!("{}/{}/audio", timestamp, file_name)
        } else {
            // Original behavior: put audio_000.mp3... under the timestamp folder
            format!("{}/audio", timestamp)
        };

        // Fresh config per file (cheap)
        let spec = writer::default_mono_24k_config(64);

        let mut mp3 = writer::Mp3Splitter::new(mp3_prefix, spec, Duration::from_hours(2))
            .context("init mp3 writer")
            .unwrap_or_log();

        let total_lines = read_non_empty_lines(&txt_path)
            .with_context(|| format!("Failed reading lines for {}", txt_path.display()))
            .unwrap_or_log();

        tracing::info!(
            "Target file {} total {} line",
            file_label,
            total_lines.len()
        );

        let sem = Arc::new(Semaphore::new(cli.concurrency * 2));
        let (tx, mut rx) = mpsc::channel::<Msg>(cli.concurrency * 2);

        let tts_engine2 = tts_engine.clone();
        let voice2 = voice;

        let producer: tokio::task::JoinHandle<anyhow::Result<()>> = tokio::spawn(async move {
            let mut set = JoinSet::<anyhow::Result<()>>::new();

            let header_span = tracing::info_span!("task");
            header_span.pb_set_style(
                &ProgressStyle::with_template("{spinner} {msg}\n{wide_bar} {pos}/{len}").unwrap(),
            );
            header_span.pb_set_length(total_lines.len() as u64);
            header_span.pb_set_message(format!("Processing {}", file_label).as_str());
            header_span
                .pb_set_finish_message(format!("All items processed ({})", file_label).as_str());

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

                let engine = tts_engine2.clone();
                let voice = voice2;

                set.spawn(async move {
                    let _permit = permit;
                    tracing::info!("Audio idx {} started", current_audio_idx);

                    let res = engine
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
                mp3.write_f32_mono(&audio)
                    .expect_or_log("Failed to write to mp3");
                tracing::info!("Audio idx {next_expected} took {:?}", took);
                next_expected += 1;
            }
        }

        producer
            .await
            .unwrap()
            .expect_or_log("Failed to finish synth task");

        mp3.finalize().expect_or_log("Failed to finalize mp3 write");

        tracing::info!("Finished {}", txt_path.display());
    }
}
