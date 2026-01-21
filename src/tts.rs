use kokoro_tts::KokoroTts;
use tokio::sync::OnceCell;
use tracing_unwrap::ResultExt;

static KOKORO_TTS: OnceCell<KokoroTts> = OnceCell::const_new();

pub async fn init_tts(tts_model: String, voice_model: String, concurrency: usize) -> &'static KokoroTts {
    KOKORO_TTS
        .get_or_init(|| async {
            KokoroTts::new_with_pool(tts_model, voice_model, concurrency)
                .await
                .expect_or_log("Failed to initialize KokoroTTS engine")
        })
        .await
}