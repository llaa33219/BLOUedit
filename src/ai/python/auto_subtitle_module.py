import os
import sys
import json
import time
import tempfile
import subprocess
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Callable

# Try to import required libraries
try:
    import torch
    import whisper
    import ffmpeg
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
    from googletrans import Translator
    import librosa
    import pysrt
    import webvtt
    from ass import Document
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install required libraries via pip:")
    print("pip install torch whisper-openai ffmpeg-python transformers googletrans==4.0.0-rc1 librosa pysrt webvtt-py pyass")
    sys.exit(1)

class AutoSubtitleEngine:
    """Engine for automatic subtitle generation from video files."""
    
    def __init__(self):
        self.whisper_model = None
        self.hf_asr_pipeline = None
        self.hf_translation_pipeline = None
        self.translator = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto Subtitle Engine initialized (using device: {self.device})")
        
    def load_whisper_model(self, model_size="medium"):
        """Load the Whisper model for ASR."""
        if self.whisper_model is None:
            print(f"Loading Whisper {model_size} model...")
            self.whisper_model = whisper.load_model(model_size, device=self.device)
            print(f"Whisper {model_size} model loaded")
        return self.whisper_model
        
    def load_hf_asr_pipeline(self):
        """Load the Hugging Face ASR pipeline."""
        if self.hf_asr_pipeline is None:
            print("Loading Hugging Face ASR pipeline...")
            model_id = "openai/whisper-large-v3"
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
            model.to(self.device)
            self.hf_asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=True,
                device=self.device
            )
            print("Hugging Face ASR pipeline loaded")
        return self.hf_asr_pipeline
        
    def load_translator(self):
        """Load the translator for subtitle translation."""
        if self.translator is None:
            print("Loading translator...")
            self.translator = Translator()
            print("Translator loaded")
        return self.translator
        
    def extract_audio(self, video_path: str, audio_path: Optional[str] = None) -> str:
        """Extract audio from video file."""
        if audio_path is None:
            # Create temporary audio file if no path is provided
            audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            
        print(f"Extracting audio from {video_path} to {audio_path}")
        
        try:
            # Use ffmpeg to extract audio
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ac=1, ar='16k')
            ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, overwrite_output=True)
            print(f"Audio extraction complete: {audio_path}")
            return audio_path
        except ffmpeg.Error as e:
            print(f"Error extracting audio: {e.stderr.decode()}")
            raise
            
    def transcribe_audio(self, audio_path: str, language: str = "auto") -> List[Dict[str, Any]]:
        """Transcribe audio file to text segments."""
        print(f"Transcribing audio: {audio_path}")
        model = self.load_whisper_model()
        
        # Transcribe audio using Whisper
        result = model.transcribe(
            audio_path,
            language=None if language == "auto" else language,
            task="transcribe",
            verbose=False
        )
        
        # Extract segments
        segments = result["segments"]
        print(f"Transcription complete: {len(segments)} segments found")
        
        return segments
        
    def translate_segments(self, segments: List[Dict[str, Any]], source_lang: str, target_lang: str) -> List[Dict[str, Any]]:
        """Translate segments to the target language."""
        translator = self.load_translator()
        
        translated_segments = []
        for segment in segments:
            # Translate the text
            try:
                translation = translator.translate(
                    segment["text"],
                    src=source_lang if source_lang != "auto" else None,
                    dest=target_lang
                )
                # Create new segment with translated text
                new_segment = segment.copy()
                new_segment["text"] = translation.text
                translated_segments.append(new_segment)
            except Exception as e:
                print(f"Translation error: {e}")
                translated_segments.append(segment)  # Keep original if translation fails
                
        return translated_segments
        
    def format_srt(self, segments: List[Dict[str, Any]], params: Dict[str, Any]) -> str:
        """Format segments as SRT subtitle file."""
        srt_content = ""
        for i, segment in enumerate(segments):
            start_time = self._format_timestamp_srt(segment["start"])
            end_time = self._format_timestamp_srt(segment["end"])
            
            # Add index
            srt_content += f"{i+1}\n"
            # Add timestamp
            srt_content += f"{start_time} --> {end_time}\n"
            # Add text
            srt_content += f"{segment['text']}"
            # Add confidence if requested
            if params.get("include_confidence", False):
                srt_content += f" [{segment.get('confidence', 0):.2f}]"
            srt_content += "\n\n"
            
        return srt_content
        
    def format_vtt(self, segments: List[Dict[str, Any]], params: Dict[str, Any]) -> str:
        """Format segments as VTT subtitle file."""
        vtt_content = "WEBVTT\n\n"
        
        for i, segment in enumerate(segments):
            start_time = self._format_timestamp_vtt(segment["start"])
            end_time = self._format_timestamp_vtt(segment["end"])
            
            # Add cue identifier
            vtt_content += f"cue-{i+1}\n"
            # Add timestamp
            vtt_content += f"{start_time} --> {end_time}\n"
            # Add text
            vtt_content += f"{segment['text']}"
            # Add confidence if requested
            if params.get("include_confidence", False):
                vtt_content += f" [{segment.get('confidence', 0):.2f}]"
            vtt_content += "\n\n"
            
        return vtt_content
        
    def format_ass(self, segments: List[Dict[str, Any]], params: Dict[str, Any]) -> str:
        """Format segments as ASS subtitle file."""
        doc = Document()
        
        # Create a style
        style = doc.styles.add_style("Default")
        style.fontname = params.get("font_name", "Arial")
        style.fontsize = params.get("font_size", 24)
        style.primary_color = params.get("font_color", "white")
        style.outline_color = params.get("outline_color", "black")
        style.back_color = params.get("background_color", "#000000AA")
        style.outline = params.get("outline_width", 1)
        style.alignment = 2  # Bottom center
        
        # Add events for each segment
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
            
            event = doc.events.add_dialogue()
            event.start = self._seconds_to_ass_time(start_time)
            event.end = self._seconds_to_ass_time(end_time)
            event.text = text
            event.style = "Default"
            
        return str(doc)
        
    def format_txt(self, segments: List[Dict[str, Any]], params: Dict[str, Any]) -> str:
        """Format segments as plain text with timestamps."""
        txt_content = ""
        
        for segment in segments:
            start_time = self._format_timestamp_txt(segment["start"])
            end_time = self._format_timestamp_txt(segment["end"])
            
            txt_content += f"[{start_time} - {end_time}] {segment['text']}\n"
            
        return txt_content
        
    def format_json(self, segments: List[Dict[str, Any]], params: Dict[str, Any]) -> str:
        """Format segments as JSON."""
        json_data = {
            "segments": segments,
            "metadata": {
                "source": params.get("input_path", ""),
                "language": params.get("language", "auto"),
                "format": "json",
                "segments_count": len(segments),
                "duration": segments[-1]["end"] if segments else 0,
                "creation_time": time.time()
            }
        }
        
        return json.dumps(json_data, indent=2)
        
    def _format_timestamp_srt(self, seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")
        
    def _format_timestamp_vtt(self, seconds: float) -> str:
        """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        
    def _format_timestamp_txt(self, seconds: float) -> str:
        """Format seconds as plain text timestamp (HH:MM:SS)."""
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
    def _seconds_to_ass_time(self, seconds: float) -> str:
        """Convert seconds to ASS timestamp format (H:MM:SS.cc)."""
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        seconds = seconds % 60
        centiseconds = int((seconds % 1) * 100)
        seconds = int(seconds)
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
    
    def process_subtitles(self, params: Dict[str, Any], progress_callback: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
        """Process video and generate subtitles."""
        result = {"success": False, "message": ""}
        
        try:
            # Extract audio if needed
            if params.get("extract_audio_only", False):
                audio_path = self.extract_audio(params["input_path"], params["extracted_audio_path"])
                result["success"] = True
                result["message"] = f"Audio extracted to {audio_path}"
                if progress_callback:
                    progress_callback(1.0)
                return result
                
            # Extract audio for processing
            if progress_callback:
                progress_callback(0.1)
                
            audio_path = self.extract_audio(params["input_path"])
            
            # Transcribe audio
            if progress_callback:
                progress_callback(0.3)
                
            segments = self.transcribe_audio(audio_path, params.get("language", "auto"))
            
            # Process segments (merge/split as needed)
            if params.get("merge_short_segments", True):
                segments = self._merge_short_segments(segments, params.get("min_segment_duration_ms", 500))
                
            if params.get("split_long_segments", True):
                segments = self._split_long_segments(segments, params.get("max_segment_duration_ms", 5000))
                
            # Apply text processing (line wrapping, etc.)
            segments = self._process_text(segments, 
                                         params.get("max_chars_per_line", 42),
                                         params.get("max_lines_per_subtitle", 2))
            
            if progress_callback:
                progress_callback(0.6)
                
            # Translate if requested
            if params.get("translation_mode", "none") != "none":
                source_lang = params.get("source_language", "auto")
                target_lang = params.get("target_language", "en")
                segments = self.translate_segments(segments, source_lang, target_lang)
                
            if progress_callback:
                progress_callback(0.8)
                
            # Format subtitles according to requested format
            subtitle_text = ""
            format_type = params.get("format", "srt")
            
            if format_type == "srt":
                subtitle_text = self.format_srt(segments, params)
            elif format_type == "vtt":
                subtitle_text = self.format_vtt(segments, params)
            elif format_type == "ass":
                subtitle_text = self.format_ass(segments, params)
            elif format_type == "txt":
                subtitle_text = self.format_txt(segments, params)
            elif format_type == "json":
                subtitle_text = self.format_json(segments, params)
            else:
                subtitle_text = self.format_srt(segments, params)  # Default to SRT
                
            # Write to output file
            with open(params["output_path"], "w", encoding="utf-8") as f:
                f.write(subtitle_text)
                
            if progress_callback:
                progress_callback(1.0)
                
            # Clean up temporary files
            if os.path.exists(audio_path) and audio_path.startswith(tempfile.gettempdir()):
                os.unlink(audio_path)
                
            result["success"] = True
            result["message"] = f"Subtitles generated successfully: {params['output_path']}"
            result["segments_count"] = len(segments)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            result["success"] = False
            result["message"] = f"Error generating subtitles: {str(e)}"
            
        return result
        
    def _merge_short_segments(self, segments: List[Dict[str, Any]], min_duration_ms: int) -> List[Dict[str, Any]]:
        """Merge segments that are too short."""
        if not segments:
            return segments
            
        min_duration = min_duration_ms / 1000.0  # Convert to seconds
        result = []
        current = segments[0].copy()
        
        for i in range(1, len(segments)):
            segment = segments[i]
            segment_duration = current["end"] - current["start"]
            
            # If current segment is too short and not the last one, merge with next
            if segment_duration < min_duration:
                current["end"] = segment["end"]
                current["text"] += " " + segment["text"]
                # Average the confidence
                if "confidence" in current and "confidence" in segment:
                    current["confidence"] = (current["confidence"] + segment["confidence"]) / 2
            else:
                result.append(current)
                current = segment.copy()
                
        # Add the last segment
        if current:
            result.append(current)
            
        return result
        
    def _split_long_segments(self, segments: List[Dict[str, Any]], max_duration_ms: int) -> List[Dict[str, Any]]:
        """Split segments that are too long."""
        if not segments:
            return segments
            
        max_duration = max_duration_ms / 1000.0  # Convert to seconds
        result = []
        
        for segment in segments:
            segment_duration = segment["end"] - segment["start"]
            
            # If segment is short enough, add it as is
            if segment_duration <= max_duration:
                result.append(segment)
            else:
                # Split into multiple segments
                num_parts = int(segment_duration / max_duration) + 1
                part_duration = segment_duration / num_parts
                
                words = segment["text"].split()
                words_per_part = len(words) // num_parts
                
                for i in range(num_parts):
                    start_idx = i * words_per_part
                    end_idx = (i + 1) * words_per_part if i < num_parts - 1 else len(words)
                    
                    part_text = " ".join(words[start_idx:end_idx])
                    part_start = segment["start"] + i * part_duration
                    part_end = segment["start"] + (i + 1) * part_duration if i < num_parts - 1 else segment["end"]
                    
                    part = segment.copy()
                    part["start"] = part_start
                    part["end"] = part_end
                    part["text"] = part_text
                    
                    result.append(part)
                    
        return result
        
    def _process_text(self, segments: List[Dict[str, Any]], max_chars_per_line: int, max_lines: int) -> List[Dict[str, Any]]:
        """Process text to respect line length constraints."""
        result = []
        
        for segment in segments:
            text = segment["text"]
            
            # If text is short enough, keep it as is
            if len(text) <= max_chars_per_line:
                result.append(segment)
                continue
                
            # Split into lines
            words = text.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line) + len(word) + 1 <= max_chars_per_line:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
                    
                    # If we have reached max lines, stop
                    if len(lines) >= max_lines - 1:
                        break
                        
            # Add the last line if not empty
            if current_line:
                lines.append(current_line)
                
            # If we have more lines than max_lines, combine the remaining words
            if len(lines) > max_lines:
                lines = lines[:max_lines-1]
                remaining_words = words[sum(len(line.split()) for line in lines):]
                if remaining_words:
                    lines.append(" ".join(remaining_words))
                    
            # Create a new segment with formatted text
            new_segment = segment.copy()
            new_segment["text"] = "\n".join(lines)
            result.append(new_segment)
            
        return result
        
    def check_requirements(self) -> Dict[str, bool]:
        """Check if all requirements for subtitle generation are met."""
        requirements = {
            "torch": False,
            "whisper": False,
            "ffmpeg": False,
            "transformers": False,
            "googletrans": False,
            "librosa": False,
            "subtitle_formats": False
        }
        
        try:
            import torch
            requirements["torch"] = True
        except ImportError:
            pass
            
        try:
            import whisper
            requirements["whisper"] = True
        except ImportError:
            pass
            
        try:
            import ffmpeg
            requirements["ffmpeg"] = True
        except ImportError:
            pass
            
        try:
            from transformers import pipeline
            requirements["transformers"] = True
        except ImportError:
            pass
            
        try:
            from googletrans import Translator
            requirements["googletrans"] = True
        except ImportError:
            pass
            
        try:
            import librosa
            requirements["librosa"] = True
        except ImportError:
            pass
            
        try:
            import pysrt
            import webvtt
            from ass import Document
            requirements["subtitle_formats"] = True
        except ImportError:
            pass
            
        return requirements 