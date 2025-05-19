import os
import sys
import time
import json
import tempfile
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Callable, Union

# Try to import required libraries
try:
    import torch
    import librosa
    import soundfile as sf
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    import audiocraft
    from audiocraft.models import MusicGen
    from audiocraft.data.audio import audio_write
    from scipy import signal
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please install required libraries via pip:")
    print("pip install torch librosa soundfile transformers audiocraft scipy")
    sys.exit(1)

class MusicGeneratorEngine:
    """Engine for AI-based music generation."""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_canceled = False
        print(f"Music Generator Engine initialized (using device: {self.device})")
        
    def load_model(self, model_name="facebook/musicgen-small"):
        """Load a music generation model."""
        if model_name in self.models:
            self.current_model = self.models[model_name]
            return self.current_model
            
        print(f"Loading music model: {model_name}")
        try:
            if "facebook/musicgen" in model_name:
                # Load MusicGen model
                model = MusicGen.get_pretrained(model_name.split('/')[-1])
                model.set_generation_params(use_sampling=True, temperature=1.0)
            else:
                # Load generic transformers model
                processor = AutoProcessor.from_pretrained(model_name)
                model = MusicgenForConditionalGeneration.from_pretrained(model_name)
                model.to(self.device)
            
            self.models[model_name] = model
            self.current_model = model
            print(f"Model {model_name} loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def generate_music(self, params: Dict[str, Any], progress_callback: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
        """Generate music based on parameters."""
        self.is_canceled = False
        result = {"success": False, "message": ""}
        
        try:
            if progress_callback:
                progress_callback(0.1)
                
            # Load appropriate model based on genre or task
            model_name = "facebook/musicgen-small"  # Default model
            if params.get("genre") == "classical":
                model_name = "facebook/musicgen-melody"
            elif params.get("genre") == "electronic" or params.get("mood") == "energetic":
                model_name = "facebook/musicgen-medium"
                
            model = self.load_model(model_name)
            if model is None:
                result["message"] = "Failed to load music generation model"
                return result
                
            if progress_callback:
                progress_callback(0.2)
                
            # Check if operation was canceled
            if self.is_canceled:
                result["message"] = "Operation canceled by user"
                return result
                
            # Prepare the generation parameters
            duration_seconds = params.get("duration_seconds", 30)
            
            # Build the prompt based on parameters
            prompt = self._build_prompt(params)
            
            if progress_callback:
                progress_callback(0.3)
                
            # Generate the music
            print(f"Generating music with prompt: {prompt}")
            
            if hasattr(model, 'set_generation_params'):  # MusicGen model
                # Set generation parameters
                model.set_generation_params(
                    duration=duration_seconds,
                    temperature=self._get_temperature_from_params(params),
                    top_k=250,
                    top_p=0.0,
                    cfg_coef=3.0
                )
                
                # Generate music
                wav = model.generate([prompt], progress=True)
                
                if progress_callback:
                    progress_callback(0.7)
                    
                # Check if operation was canceled during generation
                if self.is_canceled:
                    result["message"] = "Operation canceled by user"
                    return result
                
                # Process the generated audio
                sampling_rate = params.get("sample_rate", 44100)
                audio_data = wav[0].cpu().numpy()
                
                # Apply post-processing
                audio_data = self._post_process_audio(audio_data, params)
                
                if progress_callback:
                    progress_callback(0.8)
                
                # Save the audio file
                output_path = params.get("output_path", "generated_music.wav")
                file_format = params.get("file_format", "wav")
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                
                # Save the file
                sf.write(output_path, audio_data, sampling_rate, 
                         format=file_format if file_format != "mp3" else "WAV",
                         subtype='PCM_16' if params.get("bit_depth", 16) == 16 else 'PCM_24')
                
                # If MP3 was requested, convert from WAV
                if file_format == "mp3" and output_path.endswith(".mp3"):
                    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                    sf.write(temp_wav, audio_data, sampling_rate)
                    self._convert_to_mp3(temp_wav, output_path)
                    os.unlink(temp_wav)
            
            elif hasattr(model, 'generate'):  # Transformers model
                # Use transformers pipeline
                inputs = processor(
                    text=[prompt],
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                
                # Generate audio
                audio_values = model.generate(
                    **inputs,
                    max_length=duration_seconds * 50,  # Approximation
                    do_sample=True,
                    temperature=self._get_temperature_from_params(params)
                )
                
                if progress_callback:
                    progress_callback(0.7)
                    
                # Save audio
                sampling_rate = params.get("sample_rate", 44100)
                audio_data = audio_values[0].cpu().numpy()
                
                # Apply post-processing
                audio_data = self._post_process_audio(audio_data, params)
                
                # Save the audio file
                output_path = params.get("output_path", "generated_music.wav")
                sf.write(output_path, audio_data, sampling_rate)
            
            if progress_callback:
                progress_callback(1.0)
                
            result["success"] = True
            result["message"] = f"Music generated successfully: {output_path}"
            result["output_path"] = output_path
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            result["success"] = False
            result["message"] = f"Error generating music: {str(e)}"
            
        return result
        
    def _build_prompt(self, params: Dict[str, Any]) -> str:
        """Build a text prompt for music generation based on parameters."""
        # Start with the text prompt if provided
        prompt = params.get("text_prompt", "")
        
        # If no direct prompt is provided, build one from parameters
        if not prompt:
            genre = params.get("genre", "ambient")
            mood = params.get("mood", "calm")
            
            if genre == "custom" and params.get("custom_genre"):
                genre = params.get("custom_genre")
                
            # Build basic prompt
            prompt = f"{genre} music that sounds {mood}"
            
            # Add tempo information
            tempo = params.get("tempo_bpm", 120)
            if tempo < 80:
                prompt += ", slow tempo"
            elif tempo > 140:
                prompt += ", fast tempo"
                
            # Add instrument information
            if params.get("include_drums", True):
                prompt += " with drums"
            if params.get("include_bass", True):
                prompt += " and bass"
            if params.get("include_melody", True):
                prompt += " and melody"
                
            # Add structure info
            if params.get("intro_seconds", 0) > 5:
                prompt += ", with a long intro"
            if params.get("outro_seconds", 0) > 5:
                prompt += " and smooth outro"
                
            # If loopable is requested
            if params.get("loop_friendly", False):
                prompt += ", seamless loop"
        
        print(f"Generated prompt: {prompt}")
        return prompt
        
    def _get_temperature_from_params(self, params: Dict[str, Any]) -> float:
        """Determine generation temperature based on parameters."""
        # Higher temperature = more creativity but less coherence
        mood = params.get("mood", "calm")
        
        # Map moods to temperature values
        mood_temps = {
            "calm": 0.7,
            "happy": 0.8,
            "sad": 0.6,
            "energetic": 0.9,
            "romantic": 0.7,
            "suspenseful": 0.75,
            "epic": 0.85,
            "playful": 0.9,
            "mysterious": 0.8,
            "dark": 0.7
        }
        
        return mood_temps.get(mood, 0.8)
        
    def _post_process_audio(self, audio_data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Apply post-processing to generated audio."""
        # Apply volume adjustment
        volume = params.get("volume", 1.0)
        if volume != 1.0:
            audio_data = audio_data * volume
            
        # Apply normalization if requested
        if params.get("normalize_audio", True):
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
                
        # Make loop-friendly if requested
        if params.get("loop_friendly", False):
            # Apply crossfade between start and end
            fade_length = min(int(len(audio_data) * 0.05), 8000)  # 5% of the audio or max 8000 samples
            
            # Create fade in/out windows
            fade_in = np.linspace(0, 1, fade_length)
            fade_out = np.linspace(1, 0, fade_length)
            
            # Apply fade in at the beginning
            audio_data[:fade_length] = audio_data[:fade_length] * fade_in
            
            # Apply fade out at the end
            audio_data[-fade_length:] = audio_data[-fade_length:] * fade_out
            
            # Create a seamless loop by crossfading the beginning and end
            crossfaded = audio_data.copy()
            crossfaded[:fade_length] = audio_data[:fade_length] + audio_data[-fade_length:] * fade_in
            crossfaded[-fade_length:] = audio_data[-fade_length:] + audio_data[:fade_length] * fade_out
            
            audio_data = crossfaded
            
        return audio_data
            
    def _convert_to_mp3(self, wav_path: str, mp3_path: str, bitrate: str = "192k") -> bool:
        """Convert WAV file to MP3 using ffmpeg."""
        try:
            import subprocess
            command = [
                "ffmpeg", "-y",
                "-i", wav_path,
                "-codec:a", "libmp3lame",
                "-b:a", bitrate,
                mp3_path
            ]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except Exception as e:
            print(f"Error converting to MP3: {e}")
            return False
    
    def cancel(self) -> bool:
        """Cancel ongoing music generation."""
        self.is_canceled = True
        return True
        
    def get_available_models(self) -> List[str]:
        """Get a list of available music generation models."""
        return [
            "facebook/musicgen-small",
            "facebook/musicgen-medium",
            "facebook/musicgen-melody",
            "facebook/musicgen-large"
        ]
        
    def check_requirements(self) -> Dict[str, bool]:
        """Check if all requirements for music generation are met."""
        requirements = {
            "torch": False,
            "librosa": False,
            "soundfile": False,
            "transformers": False,
            "audiocraft": False,
            "scipy": False,
            "cuda": torch.cuda.is_available()
        }
        
        try:
            import torch
            requirements["torch"] = True
        except ImportError:
            pass
            
        try:
            import librosa
            requirements["librosa"] = True
        except ImportError:
            pass
            
        try:
            import soundfile
            requirements["soundfile"] = True
        except ImportError:
            pass
            
        try:
            from transformers import AutoProcessor
            requirements["transformers"] = True
        except ImportError:
            pass
            
        try:
            import audiocraft
            requirements["audiocraft"] = True
        except ImportError:
            pass
            
        try:
            from scipy import signal
            requirements["scipy"] = True
        except ImportError:
            pass
            
        return requirements 