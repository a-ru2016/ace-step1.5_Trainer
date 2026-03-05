import os
import sys
import json
import argparse
import shutil
import warnings
import re
import gc
import subprocess
import base64
import mimetypes
import time
import requests
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import torch
from langdetect import detect, DetectorFactory
# 言語判定の結果を固定
DetectorFactory.seed = 0

# ==========================================
# 環境・パス設定 (既存設定を継承)
# ==========================================
if sys.platform == "win32":
    venv_site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    nvidia_libs = ["cublas", "cudnn", "cuda_runtime"]
    for lib in nvidia_libs:
        bin_path = venv_site_packages / "nvidia" / lib / "bin"
        if bin_path.exists():
            os.add_dll_directory(str(bin_path))
            os.environ["PATH"] = str(bin_path) + os.pathsep + os.environ["PATH"]
            
    ffmpeg_bin_path = Path(r"C:\Users\newuser\ffmpeg-8.0.1-full_build-shared\bin") 
    if ffmpeg_bin_path.exists():
        os.add_dll_directory(str(ffmpeg_bin_path))
        os.environ["PATH"] = str(ffmpeg_bin_path) + os.pathsep + os.environ["PATH"]

warnings.filterwarnings("ignore")

def estimate_key(y, sr):
    """Estimate the musical key of the audio using chroma features."""
    try:
        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_sum = np.sum(chroma, axis=1)
        
        # Krumhansl-Schmuckler profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        best_corr = -1
        best_key = "Unknown"
        
        for i in range(12):
            # Rotate profiles
            major_rot = np.roll(major_profile, i)
            minor_rot = np.roll(minor_profile, i)
            
            # Correlation
            corr_major = np.corrcoef(chroma_sum, major_rot)[0, 1]
            corr_minor = np.corrcoef(chroma_sum, minor_rot)[0, 1]
            
            if corr_major > best_corr:
                best_corr = corr_major
                best_key = f"{keys[i]} Major"
            if corr_minor > best_corr:
                best_corr = corr_minor
                best_key = f"{keys[i]} Minor"
        
        return best_key
    except Exception:
        return "Unknown"

def generate_chord_templates():
    """Generate chroma templates for various chord types."""
    templates = {}
    chord_types = {
        'maj': [0, 4, 7],
        'min': [0, 3, 7],
        'dim': [0, 3, 6],
        'aug': [0, 4, 8],
        '7': [0, 4, 7, 10],
        'maj7': [0, 4, 7, 11],
        'min7': [0, 3, 7, 10],
        'dim7': [0, 3, 6, 9],
        'hdim7': [0, 3, 6, 10],
        'sus2': [0, 2, 7],
        'sus4': [0, 5, 7],
        'aug7': [0, 4, 8, 10]
    }
    
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for i, root in enumerate(notes):
        for chord_name, intervals in chord_types.items():
            template = np.zeros(12)
            for interval in intervals:
                template[(i + interval) % 12] = 1.0
            # Normalize template
            template = template / np.linalg.norm(template)
            
            # Name formatting
            if chord_name == 'maj':
                name = root
            elif chord_name == 'min':
                name = root + 'm'
            else:
                name = root + chord_name
            templates[name] = template
            
    return templates, notes

def estimate_chords(audio_path, structure_tags_text=None):
    """librosaを使ってビート同期のコード進行を推定する"""
    try:
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        
        # Harmonic-percussive separation (和音成分を強調)
        y_harmonic, _ = librosa.effects.hpss(y)
        
        # Beat tracking (拍を検出)
        tempo, beat_frames = librosa.beat.beat_track(y=y_harmonic, sr=sr)
        if len(beat_frames) == 0:
            return ""
            
        # Chroma extraction (CQT)
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        
        # Beat-synchronous chroma (拍ごとに集約)
        chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)
        
        # N.C.判定用: 各ビートの音の強さ(ノルム)
        # 無音やドラムしか鳴っていない箇所を判定する
        chroma_raw_norms = np.linalg.norm(chroma_sync, axis=0)
        # しきい値は経験値（音量が小さい区間をN.C.とする）
        nc_threshold = 0.5 * np.median(chroma_raw_norms) 
        
        # Bass tracking
        fmin = librosa.note_to_hz('C1')
        cqt_bass = np.abs(librosa.cqt(y, sr=sr, fmin=fmin, n_bins=36, bins_per_octave=12))
        cqt_bass_sync = librosa.util.sync(cqt_bass, beat_frames, aggregate=np.median)
        bass_notes_idx = np.argmax(cqt_bass_sync, axis=0) % 12
        
        # Template matching
        templates, notes = generate_chord_templates()
        template_names = list(templates.keys())
        template_matrix = np.array(list(templates.values())).T
        
        # Normalize chroma
        chroma_norms = np.linalg.norm(chroma_sync, axis=0)
        chroma_norms[chroma_norms == 0] = 1
        chroma_sync_norm = chroma_sync / chroma_norms
        
        # Calculate similarity
        similarities = np.dot(template_matrix.T, chroma_sync_norm)
        best_chords_idx = np.argmax(similarities, axis=0)
        
        chords = []
        for i in range(len(best_chords_idx)):
            # 音量が規定値以下ならN.C. (No Chord)
            if chroma_raw_norms[i] < nc_threshold:
                chords.append("N.C.")
                continue
                
            chord_name = template_names[best_chords_idx[i]]
            bass_note = notes[bass_notes_idx[i]]
            
            # Add bass note notation if it's different from the root (fractional chord / On chord)
            root_str = chord_name.replace('maj7', '').replace('min7', '').replace('dim7', '').replace('hdim7', '').replace('aug7', '')
            root_str = root_str.replace('m', '').replace('dim', '').replace('aug', '').replace('sus2', '').replace('sus4', '').replace('7', '')
            
            if root_str != bass_note:
                chord_name = f"{chord_name}/{bass_note}"
            
            chords.append(chord_name)
        
        # Format into bars (assuming 4 beats per bar)
        bars = []
        last_c = None
        for i in range(0, len(chords), 4):
            bar_chords = chords[i:i+4]
            clean_bar = []
            
            for c in bar_chords:
                if c == "N.C.":
                    clean_bar.append("N.C.")
                    last_c = "N.C."
                elif c != last_c:
                    clean_bar.append(c)
                    last_c = c
                else:
                    # 同じコードが続く場合は N.C. と表記する
                    clean_bar.append("N.C.")
                    
            # もし小節内がすべて N.C. ならまとめる
            if all(x == "N.C." for x in clean_bar) and len(clean_bar) > 0:
                 bars.append(["N.C."])
            else:
                 bars.append(clean_bar)
        
        # 構造タグ（[Verse 1]など）の同期
        # 小節に対するタグの割り当て
        bar_tags = {}
        if structure_tags_text:
            lines = structure_tags_text.split('\n')
            acestep_lyric_count = 0
            tags_info = []
            
            for line in lines:
                line = line.strip()
                if not line: continue
                if re.match(r'^\[.*?\]$', line):
                    tags_info.append((line, acestep_lyric_count))
                else:
                    acestep_lyric_count += 1
            
            total_bars = len(bars)
            if acestep_lyric_count > 0 and total_bars > 0:
                for tag, a_idx in tags_info:
                    target_bar_idx = int((a_idx / acestep_lyric_count) * total_bars)
                    if target_bar_idx >= total_bars:
                        target_bar_idx = total_bars - 1
                    
                    if target_bar_idx not in bar_tags:
                        bar_tags[target_bar_idx] = []
                    bar_tags[target_bar_idx].append(tag)
        
        # 出力文字列の組み立て
        progression_str = []
        for i, bar in enumerate(bars):
            if i in bar_tags:
                for tag in bar_tags[i]:
                    progression_str.append(f"\n{tag}")
            progression_str.append("| " + " ".join(bar) + " |")
        
        return " ".join(progression_str).strip()
    except Exception as e:
        print(f"    ⚠️ Chord estimation failed: {e}")
        return ""

def analyze_audio_task(audio_path: Path):
    try:
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        duration = len(y) / sr
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = round(float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo), 1)
        
        # Key estimation
        key = estimate_key(y, sr)
        
        return {"status": "ok", "bpm": bpm, "duration": duration, "key": key}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ==========================================
# コア・パイプライン
# ==========================================
class HybridMusicPreprocessor:
    def __init__(self, music_dir: str, language: str):
        self.music_dir = Path(music_dir)
        self.language = language
        self.audio_files = [Path(r)/f for r,_,fs in os.walk(music_dir) 
                            for f in fs if Path(f).suffix.lower() in {".wav", ".mp3", ".flac", ".m4a"}]
        self.temp_dir = Path("temp_processing")
        if self.temp_dir.exists(): shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir()
        
    def clear_vram(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_acestep_model(self, model_id: str, prompt: str, extension: str):
        """ACE-Stepモデルを4bitでロードし、全曲処理後に破棄する"""
        print(f"\n📦 Loading {model_id} (4-bit)...")
        from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration, BitsAndBytesConfig
        import torch
        import librosa
        
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            ),
            device_map="auto",
            trust_remote_code=True
        ).eval()

        # Qwen2.5-Omniが内部で期待する公式システムプロンプト（Warning回避用）
        sys_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

        for i, path in enumerate(self.audio_files):
            out_path = path.parent / f"{path.stem}{extension}"
            if out_path.exists(): continue
            
            print(f"  [{i+1}/{len(self.audio_files)}] Inferring {model_id.split('-')[-1]} for: {path.name}")
            # サンプリングレートはプロセッサから取得
            audio, _ = librosa.load(str(path), sr=processor.feature_extractor.sampling_rate)
            
            conversation = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": "local_audio"}, 
                    {"type": "text", "text": prompt}
                ]}
            ]
            text_input = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            
            # ⚠️ 修正: `audios` ではなく `audio` 引数を使用
            inputs = processor(text=text_input, audio=[audio], return_tensors="pt", padding=True).to(model.device)
            
            with torch.no_grad():
                generate_ids = model.generate(**inputs, max_new_tokens=2048)
                
            # ⚠️ 修正: Tupleで返ってきた場合、最初の要素（テキストID）だけを抽出
            if isinstance(generate_ids, tuple):
                text_ids = generate_ids[0]
            else:
                text_ids = generate_ids
                
            # 入力プロンプト部分を切り落としてデコード
            text_ids = text_ids[:, inputs.input_ids.size(1):]
            result = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
            out_path.write_text(result, encoding="utf-8")

        del model, processor
        self.clear_vram()

    # ==========================================
    # Gemini API: キャプション + 構造タグ 同時生成
    # ==========================================
    GEMINI_PROMPT = """Analyze the input audio to generate a detailed caption and structured lyrics.
The caption should be a detailed description of the audio in English (genre, instruments, vocals, mood, tempo, structure etc.).
The lyrics should contain ONLY structural section tags like [Intro], [Verse 1], [Pre-Chorus], [Chorus], [Bridge], [Guitar Solo], [Outro] etc.
Do NOT transcribe actual sung words - only provide the structural tags.
If the audio has instrumental sections, label them appropriately (e.g. [Guitar Solo], [Instrumental Break]).

**Output Format (JSON only, no markdown):**
{
    "caption": "<detailed English description of the audio>",
    "lyrics": "[Intro]\\n\\n[Verse 1]\\n\\n[Chorus]\\n..."
}"""

    def _get_audio_mime_type(self, file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            ext = Path(file_path).suffix.lower()
            mime_map = {
                ".mp3": "audio/mp3", ".wav": "audio/wav",
                ".flac": "audio/flac", ".m4a": "audio/mp4",
                ".ogg": "audio/ogg", ".aac": "audio/aac",
            }
            mime_type = mime_map.get(ext, "application/octet-stream")
        return mime_type

    def _gemini_analyze_audio(self, api_key: str, audio_path: Path, max_retries: int = 3) -> dict:
        """Gemini APIで音声を分析し、caption+lyricsのdictを返す"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}

        # 音声をbase64エンコード
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        mime_type = self._get_audio_mime_type(str(audio_path))

        body = {
            "contents": [{"role": "user", "parts": [
                {"text": self.GEMINI_PROMPT},
                {"inline_data": {"mime_type": mime_type, "data": audio_b64}}
            ]}],
            "generationConfig": {"responseMimeType": "application/json"}
        }

        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=300)
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    print(f"    ⏳ Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code != 200:
                    print(f"    ❌ Gemini API error (HTTP {resp.status_code}): {resp.text[:200]}")
                    continue

                result = resp.json()
                text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                # JSONパース
                text = text.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(text)
                return parsed
            except json.JSONDecodeError:
                # JSON抽出のフォールバック
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    try:
                        return json.loads(text[start:end+1])
                    except json.JSONDecodeError:
                        pass
                print(f"    ⚠️ JSON parse failed (attempt {attempt+1}), retrying...")
            except Exception as e:
                print(f"    ⚠️ Error (attempt {attempt+1}): {e}")
            time.sleep(1)

        return {"caption": "", "lyrics": ""}

    def run_gemini_caption_and_structure(self, api_key: str):
        """Gemini APIでキャプションと構造タグを同時に生成（Phase 1+2統合）"""
        print(f"\n🌐 Gemini API: Generating captions and structure tags...")

        for i, path in enumerate(self.audio_files):
            caption_out = path.parent / f"{path.stem}.caption.txt"
            structure_out = path.parent / f"{path.stem}.acestep_raw.txt"
            final_lyrics = path.parent / f"{path.stem}.lyrics.txt"

            # 既存の .lyrics.txt があれば構造タグ生成は不要
            need_caption = not caption_out.exists()
            need_structure = not structure_out.exists() and not final_lyrics.exists()

            if not need_caption and not need_structure:
                continue

            print(f"  [{i+1}/{len(self.audio_files)}] Analyzing: {path.name}")
            result = self._gemini_analyze_audio(api_key, path)

            caption = result.get("caption", "")
            lyrics = result.get("lyrics", "")

            if caption and not caption_out.exists():
                # コード進行を推定してキャプション末尾に追加
                print(f"    🎸 Estimating chord progression...")
                progression = estimate_chords(path, lyrics)
                if progression:
                    caption += f"\n\n[Chord Progression]\n{progression}"
                caption_out.write_text(caption, encoding="utf-8")

            if lyrics and not structure_out.exists():
                structure_out.write_text(lyrics, encoding="utf-8")

            if not caption and not lyrics:
                print(f"    ⚠️ No result for {path.name}")

    # ==========================================
    # ACE-Step 内蔵 LLM: GUIと同じ方式でキャプション+構造タグ生成
    # ==========================================
    def run_acestep_llm_labeling(self):
        """ACE-Step内蔵のDiT+LLMハンドラーでキャプション・構造タグ・メタデータを生成（GUIと同じ方式）"""
        print(f"\n🎵 ACE-Step LLM: Generating captions and structure tags (same as GUI)...")
        
        # プロジェクトルートを推定（このスクリプトが置かれているディレクトリ）
        project_root = str(Path(__file__).resolve().parent)
        checkpoint_dir = os.path.join(project_root, "checkpoints")
        config_path = os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-sft")
        lm_model_path = os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")
        lm_backend = os.environ.get("ACESTEP_LM_BACKEND", "vllm")
        
        # ACE-Stepのハンドラーをインポート・初期化
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        from acestep.gpu_config import get_gpu_config, VRAM_AUTO_OFFLOAD_THRESHOLD_GB
        
        # GPU設定を取得（GUIと同じ自動検出ロジック）
        gpu_config = get_gpu_config()
        gpu_memory_gb = gpu_config.gpu_memory_gb
        offload_to_cpu = gpu_memory_gb > 0 and gpu_memory_gb < VRAM_AUTO_OFFLOAD_THRESHOLD_GB
        offload_dit_to_cpu = gpu_config.offload_dit_to_cpu_default
        quantization = "int8_weight_only" if gpu_config.quantization_default else None
        
        print(f"  🖥️ GPU: {gpu_memory_gb:.1f}GB | offload_to_cpu={offload_to_cpu} | quantization={quantization}")
        
        # DiTハンドラー初期化（GUIと同じパラメータ）
        print("  📦 Loading DiT handler...")
        dit_handler = AceStepHandler()
        
        # Flash attention自動検出
        use_flash_attention = dit_handler.is_flash_attention_available("auto") if hasattr(dit_handler, 'is_flash_attention_available') else False
        
        # compile_model: quantization使用時はTrue必須
        compile_model = os.environ.get("ACESTEP_COMPILE_MODEL", "").strip().lower() in {"1", "true", "yes", "y", "on"}
        if quantization and not compile_model:
            compile_model = True
        
        status, success = dit_handler.initialize_service(
            project_root=project_root,
            config_path=config_path,
            device="auto",
            use_flash_attention=use_flash_attention,
            compile_model=compile_model,
            offload_to_cpu=offload_to_cpu,
            offload_dit_to_cpu=offload_dit_to_cpu,
            quantization=quantization,
        )
        if not success:
            print(f"  ❌ DiT handler initialization failed: {status}")
            return
        print(f"  ✅ DiT handler loaded")
        
        # LLMハンドラー初期化（GUIと同じパラメータ）
        print(f"  📦 Loading LLM handler (5Hz LM: {lm_model_path}, backend: {lm_backend})...")
        llm_handler = LLMHandler()
        llm_status, llm_success = llm_handler.initialize(
            checkpoint_dir=checkpoint_dir,
            lm_model_path=lm_model_path,
            backend=lm_backend,
            device="auto",
            offload_to_cpu=offload_to_cpu,
        )
        if not llm_success:
            print(f"  ❌ LLM handler initialization failed: {llm_status}")
            del dit_handler
            self.clear_vram()
            return
        print(f"  ✅ LLM handler loaded")
        
        for i, path in enumerate(self.audio_files):
            caption_out = path.parent / f"{path.stem}.caption.txt"
            structure_out = path.parent / f"{path.stem}.acestep_raw.txt"
            final_lyrics = path.parent / f"{path.stem}.lyrics.txt"
            
            # 既存の .lyrics.txt があれば構造タグ生成は不要
            need_caption = not caption_out.exists()
            need_structure = not structure_out.exists() and not final_lyrics.exists()

            if not need_caption and not need_structure:
                continue
            
            print(f"  [{i+1}/{len(self.audio_files)}] Analyzing: {path.name}")
            
            try:
                # Step 1: DiTで音声をaudio codesに変換
                audio_codes = dit_handler.convert_src_audio_to_codes(str(path))
                if not audio_codes or audio_codes.startswith("❌"):
                    print(f"    ❌ Failed to encode audio: {path.name}")
                    continue
                
                # Step 2: LLMでaudio codesからメタデータ+歌詞を生成
                metadata, status = llm_handler.understand_audio_from_codes(
                    audio_codes=audio_codes,
                    temperature=0.7,
                    use_constrained_decoding=True,
                )
                
                if not metadata:
                    print(f"    ❌ LLM labeling failed: {status}")
                    continue
                
                # キャプション保存
                caption = metadata.get("caption", "")
                lyrics = metadata.get("lyrics", "")
                
                if caption and not caption_out.exists():
                    # コード進行を推定してキャプション末尾に追加
                    print(f"    🎸 Estimating chord progression...")
                    progression = estimate_chords(path, lyrics)
                    if progression:
                        caption += f"\n\n[Chord Progression]\n{progression}"
                    caption_out.write_text(caption, encoding="utf-8")
                
                # 構造タグ（歌詞）保存
                if lyrics and not structure_out.exists():
                    structure_out.write_text(lyrics, encoding="utf-8")
                
                print(f"    ✅ caption={len(caption)}chars, lyrics={len(lyrics)}chars")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        # VRAM解放
        del dit_handler, llm_handler
        self.clear_vram()

    def apply_noise_gate(self, audio_path: Path, top_db=38):
        y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        intervals = librosa.effects.split(y, top_db=top_db, frame_length=2048, hop_length=512)
        y_gated = np.zeros_like(y)
        for start, end in intervals:
            y_gated[start:end] = y[start:end]
        gated_path = self.temp_dir / f"{audio_path.stem}_gated.wav"
        sf.write(str(gated_path), y_gated, sr)
        return gated_path

    def process_whisper_pipeline(self):
        """Demucsをサブプロセスで隔離実行し、Faster-Whisperで完璧な歌詞を抽出"""
        print("\n📦 Loading Faster-Whisper large-v3...")
        from faster_whisper import WhisperModel
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        
        blacklist = ["ご視聴ありがとうございました", "チャンネル登録", "高評価", "字幕:", "サブタイトル"]

        for i, path in enumerate(self.audio_files):
            whisper_out = path.parent / f"{path.stem}.whisper_raw.json"
            final_lyrics = path.parent / f"{path.stem}.lyrics.txt"
            
            if whisper_out.exists() or final_lyrics.exists(): continue
            
            print(f"  [{i+1}/{len(self.audio_files)}] Extracting pure lyrics for: {path.name}")
            
            # VRAMリーク防止のためDemucsを独立プロセスで実行
            # 特殊Unicode文字（⧸等）がファイル名に含まれるとDemucsが失敗するため、
            # 安全な一時ファイル名にコピーしてリトライする
            demucs_input = path
            temp_copy = None
            try:
                subprocess.run([
                    sys.executable, "-m", "demucs.separate", "--two-stems", "vocals", "-n", "htdemucs_ft", 
                    "-d", "cuda", str(path), "-o", str(self.temp_dir)
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e1:
                print(f"    ⚠️ First demucs attempt failed: {e1}")
                # 安全なASCIIファイル名にコピーしてリトライ
                safe_name = f"_demucs_temp_{i}{path.suffix}"
                temp_copy = self.temp_dir / safe_name
                shutil.copy2(str(path), str(temp_copy))
                demucs_input = temp_copy
                try:
                    subprocess.run([
                        sys.executable, "-m", "demucs.separate", "--two-stems", "vocals", "-n", "htdemucs_ft",
                        "-d", "cuda", str(temp_copy), "-o", str(self.temp_dir)
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e2:
                    print(f"    ⚠️ Demucs failed even with safe filename, skipping: {e2}")
                    continue
            
            vocal_stem = demucs_input.stem
            vocal_file = self.temp_dir / "htdemucs_ft" / vocal_stem / "vocals.wav"
            if not vocal_file.exists(): vocal_file = vocal_file.with_suffix(".mp3")
            
            if not vocal_file.exists():
                print(f"    ⚠️ Vocal file not found, skipping: {vocal_file}")
                continue
            
            gated_file = self.apply_noise_gate(vocal_file)
            
            segments, info = model.transcribe(
                str(gated_file), language=self.language, beam_size=5, condition_on_previous_text=False,
                vad_filter=True, vad_parameters=dict(min_silence_duration_ms=600, speech_pad_ms=400, threshold=0.15)
            )
            
            detected_lang = info.language
            
            valid_segments = []
            full_text_for_detection = ""
            for s in segments:
                text = s.text.strip()
                if not text or any(p in text for p in blacklist): continue
                valid_segments.append({"start": s.start, "end": s.end, "text": text})
                full_text_for_detection += text + " "
                
            # 保存データに言語情報も含める
            whisper_data = {
                "detected_language": detected_lang,
                "segments": valid_segments
            }
            whisper_out.write_text(json.dumps(whisper_data, ensure_ascii=False), encoding="utf-8")
            
            # 一時コピーの削除
            if temp_copy and temp_copy.exists():
                temp_copy.unlink()
            
        del model
        self.clear_vram()

    def proportional_structure_alignment(self, acestep_text: str, whisper_segments: list) -> str:
        """【ブレイクスルー】ACE-Stepの構造タグとWhisperの歌詞を比率計算で融合するアルゴリズム"""
        lines = acestep_text.split('\n')
        tags_info = [] 
        acestep_lyric_count = 0
        
        # ACE-Stepの幻覚テキストから「タグ」と「歌詞だったはずの行数」を解析
        for line in lines:
            line = line.strip()
            if not line: continue
            if re.match(r'^\[.*?\]$', line):
                tags_info.append((line, acestep_lyric_count))
            else:
                acestep_lyric_count += 1
                
        whisper_lines = [seg['text'] for seg in whisper_segments]
        whisper_total = len(whisper_lines)
        final_output = []
        w_idx = 0
        
        # 歌詞が存在しない、または抽出失敗時のフォールバック
        if acestep_lyric_count == 0:
            for tag, _ in tags_info: final_output.append(f"\n{tag}")
            final_output.extend(whisper_lines)
        else:
            # タグの相対的な出現位置を計算し、正確なWhisperの歌詞の間に流し込む
            for tag, a_idx in tags_info:
                target_w_idx = int((a_idx / acestep_lyric_count) * whisper_total)
                while w_idx < target_w_idx and w_idx < whisper_total:
                    final_output.append(whisper_lines[w_idx])
                    w_idx += 1
                final_output.append(f"\n{tag}")
                
            # 残りの歌詞を吐き出す
            while w_idx < whisper_total:
                final_output.append(whisper_lines[w_idx])
                w_idx += 1
                
        # 余分な改行を整理
        final_text = "\n".join(final_output).strip()
        return re.sub(r'\n{3,}', '\n\n', final_text)

    def run_pipeline(self, gemini_api_key: str = None):
        print(f"🚀 Found {len(self.audio_files)} files. Starting Hybrid Pipeline.")

        # 事前チェック: どの処理が必要かを表示
        stats = {"caption": 0, "structure": 0, "whisper": 0, "json": 0}
        for path in self.audio_files:
            caption_out = path.parent / f"{path.stem}.caption.txt"
            structure_out = path.parent / f"{path.stem}.acestep_raw.txt"
            whisper_out = path.parent / f"{path.stem}.whisper_raw.json"
            final_lyrics = path.parent / f"{path.stem}.lyrics.txt"
            json_file = path.parent / f"{path.stem}.json"

            if not caption_out.exists(): stats["caption"] += 1
            if not structure_out.exists() and not final_lyrics.exists(): stats["structure"] += 1
            if not whisper_out.exists() and not final_lyrics.exists(): stats["whisper"] += 1
            
            # JSONの欠損チェック
            if not json_file.exists():
                stats["json"] += 1
            else:
                try:
                    meta = json.loads(json_file.read_text(encoding="utf-8"))
                    if "key" not in meta or meta.get("key") == "Unknown" or "language" not in meta:
                        stats["json"] += 1
                except: stats["json"] += 1

        print(f"📊 Pre-processing Summary:")
        print(f"  - Need Captions: {stats['caption']}")
        print(f"  - Need Structure Tags: {stats['structure']}")
        print(f"  - Need Whisper Transcription: {stats['whisper']}")
        print(f"  - Need Metadata updates: {stats['json']}")
        print(f"------------------------------------------")

        # Phase 1+2: キャプション + 構造タグ生成
        if gemini_api_key:
            # オプション: Gemini APIを使用
            self.run_gemini_caption_and_structure(gemini_api_key)
        else:
            # デフォルト: ACE-Step内蔵LLM（GUIと同じ方式）
            self.run_acestep_llm_labeling()
        
        self.process_whisper_pipeline()
        
        # Phase 4: Merge & Metadata (構造タグと正確な歌詞の合成)
        print("\n🧬 Merging Structures and Lyrics (Proportional Alignment)...")
        for path in self.audio_files:
            acestep_raw = path.parent / f"{path.stem}.acestep_raw.txt"
            whisper_raw = path.parent / f"{path.stem}.whisper_raw.json"
            json_file = path.parent / f"{path.stem}.json"
            
            # 既存のJSONファイルのチェック（キーのタグ付けが漏れている場合への対応）
            if json_file.exists():
                try:
                    meta = json.loads(json_file.read_text(encoding="utf-8"))
                    updated = False
                    
                    # キー情報の欠落チェック
                    if "key" not in meta or meta.get("key") == "Unknown":
                        print(f"  🔍 Adding missing key info to: {path.name}")
                        analysis = analyze_audio_task(path)
                        if analysis.get("status") == "ok":
                            meta["key"] = analysis.get("key", "Unknown")
                            updated = True
                            print(f"    ✅ Added key: {meta['key']}")
                    
                    # 言語タグの欠落チェック（レトロフィット）
                    if "language" not in meta:
                        lyrics_path = path.parent / f"{path.stem}.lyrics.txt"
                        if lyrics_path.exists():
                            lyrics_text = lyrics_path.read_text(encoding="utf-8")
                            # 構造タグを除去して言語判定
                            plain_text = re.sub(r'\[.*?\]', '', lyrics_text).strip()
                            if plain_text:
                                try:
                                    lang = detect(plain_text)
                                    meta["language"] = lang
                                    updated = True
                                    print(f"    ✅ Retrofitted language for {path.name}: {lang}")
                                except Exception as e:
                                    print(f"    ⚠️ Language detection failed for {path.name}: {e}")

                    if updated:
                        json_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception as e:
                    print(f"    ⚠️ Error updating JSON for {path.name}: {e}")

            # 独自の合成アルゴリズムを実行
            if acestep_raw.exists() and whisper_raw.exists():
                acestep_text = acestep_raw.read_text(encoding="utf-8")
                whisper_json = json.loads(whisper_raw.read_text(encoding="utf-8"))
                
                # whisper_raw.json の形式変更に対応
                if isinstance(whisper_json, dict) and "segments" in whisper_json:
                    whisper_segs = whisper_json["segments"]
                    detected_lang = whisper_json.get("detected_language", "unknown")
                else:
                    whisper_segs = whisper_json
                    detected_lang = "unknown"
                
                final_lyrics = self.proportional_structure_alignment(acestep_text, whisper_segs)
                (path.parent / f"{path.stem}.lyrics.txt").write_text(final_lyrics, encoding="utf-8")
                
                # メタデータ生成
                analysis = analyze_audio_task(path)
                match = re.search(r"(?:\d+\s*-\s*)?(.*?)\s*\[.*\]", path.name)
                song_title = match.group(1).strip() if match and match.group(1).strip() else "楽曲"
                
                # 言語判定（Whisperの結果を優先、なければ合成後の歌詞から判定）
                if detected_lang == "unknown" or not detected_lang:
                    try:
                        plain_lyrics = re.sub(r'\[.*?\]', '', final_lyrics).strip()
                        if plain_lyrics:
                            detected_lang = detect(plain_lyrics)
                    except:
                        detected_lang = self.language # フォールバック

                model_label = "Gemini" if gemini_api_key else "ACE-Step LLM"
                meta = {
                    "title_extracted": song_title,
                    "bpm": analysis.get("bpm", 0), 
                    "duration": round(analysis.get("duration", 0), 2),
                    "key": analysis.get("key", "Unknown"),
                    "language": detected_lang,
                    "model": f"Hybrid ({model_label} Structure + Whisper_v3_Lyrics)"
                }
                (path.parent / f"{path.stem}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                
            # 中間ファイルのクリーンアップ（学習に不要。final_lyricsが存在する場合や不完全な場合も削除）
            if (path.parent / f"{path.stem}.lyrics.txt").exists():
                if acestep_raw.exists(): acestep_raw.unlink()
                if whisper_raw.exists(): whisper_raw.unlink()

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
        print("\n✅ All processing completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="音楽データセット前処理パイプライン (ACE-Step LLM + Whisper)")
    parser.add_argument("music_dir", help="音楽ファイルのディレクトリ")
    parser.add_argument("--language", default="ja", help="Whisper抽出用の言語設定")
    parser.add_argument("--gemini-api-key", default=None,
                        help="Gemini APIを使う場合のAPI key (未指定時はACE-Step内蔵LLMを使用)")
    args = parser.parse_args()

    # APIキー解決: 引数 > 環境変数
    api_key = args.gemini_api_key
    if api_key and api_key.lower() in ["none", "null", "false", ""]:
        api_key = None
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    if api_key and api_key.lower() in ["none", "null", "false", ""]:
        api_key = None

    pipeline = HybridMusicPreprocessor(args.music_dir, args.language)
    pipeline.run_pipeline(gemini_api_key=api_key)