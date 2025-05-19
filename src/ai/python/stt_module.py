#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlouEdit Speech-to-Text Module
Provides functionality for converting speech to text using various STT engines.
"""

import os
import sys
import logging
import tempfile
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("stt_module")

# Try importing required dependencies
try:
    import numpy as np
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("scipy package not found. Some audio processing may be limited.")
    SCIPY_AVAILABLE = False

# Try importing STT engines
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    logger.warning("whisper package not found. Whisper STT will be unavailable.")
    WHISPER_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    logger.warning("speech_recognition package not found. Google STT will be unavailable.")
    SR_AVAILABLE = False

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_STT_AVAILABLE = os.environ.get("AZURE_SPEECH_KEY") is not None
except ImportError:
    logger.warning("azure.cognitiveservices.speech package not found. Azure STT will be unavailable.")
    AZURE_STT_AVAILABLE = False

try:
    from vosk import Model, KaldiRecognizer, SetLogLevel
    import wave
    VOSK_AVAILABLE = True
    # Set log level to suppress Vosk debug messages
    SetLogLevel(-1)
except ImportError:
    logger.warning("vosk package not found. Offline VOSK recognition will be unavailable.")
    VOSK_AVAILABLE = False

class SpeechToTextEngine:
    """Engine for converting speech to text using various backends."""
    
    def __init__(self):
        """Initialize the STT engine."""
        self.whisper_service = WHISPER_AVAILABLE
        self.google_service = SR_AVAILABLE
        self.azure_service = AZURE_STT_AVAILABLE
        self.vosk_service = VOSK_AVAILABLE
        
        # Initialize whisper model cache
        self.whisper_models = {}
        
        # Initialize vosk model cache
        self.vosk_models = {}
    
    def transcribe(self, params):
        """
        Transcribe speech from an audio file.
        
        Args:
            params (dict): Dictionary containing transcription parameters
                - audio_path (str): Path to the audio file
                - language (str): Language code (empty for auto-detection)
                - add_timestamps (bool): Whether to add word-level timestamps
                - add_punctuation (bool): Whether to add punctuation
                - engine (str): Engine to use ('whisper', 'google', 'azure', 'vosk')
            
        Returns:
            dict: Dictionary containing transcription results
                - text (str): Full transcribed text
                - segments (list): List of segment dictionaries with:
                    - text (str): Text of the segment
                    - start (float): Start time in seconds
                    - end (float): End time in seconds
                    - confidence (float): Confidence score (0-1)
        """
        try:
            # Extract parameters
            audio_path = params.get('audio_path', '')
            language = params.get('language', '')
            add_timestamps = params.get('add_timestamps', False)
            add_punctuation = params.get('add_punctuation', True)
            engine = params.get('engine', 'whisper')
            
            if not audio_path or not os.path.exists(audio_path):
                logger.error(f"Audio file not found: {audio_path}")
                return {
                    'text': '',
                    'segments': []
                }
            
            # Determine which engine to use
            if engine == 'whisper' and self.whisper_service:
                return self._transcribe_with_whisper(audio_path, language, add_timestamps, add_punctuation)
            elif engine == 'google' and self.google_service:
                return self._transcribe_with_google(audio_path, language, add_timestamps, add_punctuation)
            elif engine == 'azure' and self.azure_service:
                return self._transcribe_with_azure(audio_path, language, add_timestamps, add_punctuation)
            elif engine == 'vosk' and self.vosk_service:
                return self._transcribe_with_vosk(audio_path, language, add_timestamps, add_punctuation)
            else:
                # Fall back to best available engine
                if self.whisper_service:
                    logger.info(f"Using whisper as fallback engine")
                    return self._transcribe_with_whisper(audio_path, language, add_timestamps, add_punctuation)
                elif self.vosk_service:
                    logger.info(f"Using vosk as fallback engine")
                    return self._transcribe_with_vosk(audio_path, language, add_timestamps, add_punctuation)
                elif self.google_service:
                    logger.info(f"Using google as fallback engine")
                    return self._transcribe_with_google(audio_path, language, add_timestamps, add_punctuation)
                elif self.azure_service:
                    logger.info(f"Using azure as fallback engine")
                    return self._transcribe_with_azure(audio_path, language, add_timestamps, add_punctuation)
                else:
                    logger.error("No STT engine available")
                    return {
                        'text': '',
                        'segments': []
                    }
        
        except Exception as e:
            logger.error(f"Failed to transcribe speech: {str(e)}")
            return {
                'text': '',
                'segments': []
            }
    
    def _transcribe_with_whisper(self, audio_path, language, add_timestamps, add_punctuation):
        """Transcribe using OpenAI's Whisper."""
        if not self.whisper_service:
            logger.error("Whisper not available")
            return {'text': '', 'segments': []}
        
        try:
            # Get or load model (use 'base' model by default for balance of accuracy and performance)
            model_name = 'base'
            if model_name not in self.whisper_models:
                logger.info(f"Loading whisper model: {model_name}")
                self.whisper_models[model_name] = whisper.load_model(model_name)
            
            model = self.whisper_models[model_name]
            
            # Prepare transcription options
            options = {}
            if language:
                options['language'] = language
            
            # Transcribe
            logger.info(f"Transcribing with whisper: {audio_path}")
            result = model.transcribe(
                audio_path, 
                verbose=False,
                word_timestamps=add_timestamps,
                **options
            )
            
            # Extract text and segments
            full_text = result.get('text', '')
            segments = []
            
            for seg in result.get('segments', []):
                segment = {
                    'text': seg.get('text', ''),
                    'start': seg.get('start', 0),
                    'end': seg.get('end', 0),
                    'confidence': 1.0  # Whisper doesn't provide confidence scores
                }
                segments.append(segment)
            
            return {
                'text': full_text,
                'segments': segments
            }
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {str(e)}")
            return {'text': '', 'segments': []}
    
    def _transcribe_with_google(self, audio_path, language, add_timestamps, add_punctuation):
        """Transcribe using Google Speech Recognition."""
        if not self.google_service:
            logger.error("Google Speech Recognition not available")
            return {'text': '', 'segments': []}
        
        try:
            # Initialize recognizer
            recognizer = sr.Recognizer()
            
            # Load audio file
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
            
            # Prepare language
            lang_code = language if language else 'en-US'
            
            # Recognize speech
            logger.info(f"Transcribing with Google: {audio_path}")
            text = recognizer.recognize_google(
                audio_data,
                language=lang_code,
                show_all=True  # Get detailed results
            )
            
            # Extract results
            full_text = ''
            segments = []
            
            if isinstance(text, dict) and 'alternative' in text:
                # Get best alternative
                best_alt = text['alternative'][0] if text['alternative'] else {}
                full_text = best_alt.get('transcript', '')
                
                # Google doesn't provide segments, so create one for the whole audio
                if full_text:
                    segments.append({
                        'text': full_text,
                        'start': 0,
                        'end': 10000, # Dummy end time
                        'confidence': best_alt.get('confidence', 1.0)
                    })
            
            return {
                'text': full_text,
                'segments': segments
            }
            
        except Exception as e:
            logger.error(f"Google transcription error: {str(e)}")
            return {'text': '', 'segments': []}
    
    def _transcribe_with_azure(self, audio_path, language, add_timestamps, add_punctuation):
        """Transcribe using Azure Speech Service."""
        if not self.azure_service:
            logger.error("Azure Speech Service not available")
            return {'text': '', 'segments': []}
        
        try:
            # Get Azure credentials
            subscription_key = os.environ.get("AZURE_SPEECH_KEY")
            region = os.environ.get("AZURE_SPEECH_REGION", "eastus")
            
            # Create speech config
            speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
            
            # Set language if specified
            if language:
                speech_config.speech_recognition_language = language
            
            # Create audio config
            audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
            
            # Create recognizer
            if add_timestamps:
                # Use detailed output format to get timestamps
                speech_config.request_word_level_timestamps()
                speech_config.output_format = speechsdk.OutputFormat.Detailed
            
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, 
                audio_config=audio_config
            )
            
            # Prepare result containers
            full_text = ""
            segments = []
            
            # Define callback for recognized event
            def recognized_callback(evt):
                nonlocal full_text
                result = evt.result
                
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    offset = result.offset / 10000000  # Convert 100ns to seconds
                    duration = result.duration.total_seconds()
                    
                    segment = {
                        'text': result.text,
                        'start': offset,
                        'end': offset + duration,
                        'confidence': 1.0  # Default confidence
                    }
                    
                    segments.append(segment)
                    full_text += result.text + " "
            
            # Connect callback
            recognizer.recognized.connect(recognized_callback)
            
            # Start recognition
            logger.info(f"Transcribing with Azure: {audio_path}")
            result = recognizer.recognize_once_async().get()
            
            # If no segments were created from callback, create one from the result
            if not segments and result.text:
                full_text = result.text
                segments.append({
                    'text': result.text,
                    'start': 0,
                    'end': 10,  # Placeholder duration
                    'confidence': 1.0
                })
            
            return {
                'text': full_text.strip(),
                'segments': segments
            }
            
        except Exception as e:
            logger.error(f"Azure transcription error: {str(e)}")
            return {'text': '', 'segments': []}
    
    def _transcribe_with_vosk(self, audio_path, language, add_timestamps, add_punctuation):
        """Transcribe using VOSK offline speech recognition."""
        if not self.vosk_service:
            logger.error("VOSK not available")
            return {'text': '', 'segments': []}
        
        try:
            # Determine model path and language
            lang_code = language if language else 'en'
            model_name = f"vosk-model-{lang_code}"
            
            # Path to models - prefer models in the local directory
            model_path = None
            local_model_path = os.path.join("models", model_name)
            user_model_path = os.path.expanduser(f"~/.cache/blouedit/models/{model_name}")
            
            if os.path.exists(local_model_path):
                model_path = local_model_path
            elif os.path.exists(user_model_path):
                model_path = user_model_path
            else:
                logger.error(f"VOSK model not found for language: {lang_code}")
                return {'text': '', 'segments': []}
            
            # Load model or get from cache
            if model_path not in self.vosk_models:
                logger.info(f"Loading VOSK model: {model_path}")
                self.vosk_models[model_path] = Model(model_path)
            
            model = self.vosk_models[model_path]
            
            # Open audio file
            with wave.open(audio_path, 'rb') as wf:
                # Create recognizer
                rec = KaldiRecognizer(model, wf.getframerate())
                rec.SetWords(add_timestamps)
                
                # Process audio in chunks
                full_text = ""
                segments = []
                chunk_size = 4000  # Adjust for memory vs speed trade-off
                
                while True:
                    data = wf.readframes(chunk_size)
                    if len(data) == 0:
                        break
                    
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        
                        if 'text' in result and result['text']:
                            text = result['text']
                            full_text += text + " "
                            
                            # Create segment
                            segment = {
                                'text': text,
                                # Vosk doesn't provide accurate timestamps in chunk mode
                                'start': 0,
                                'end': 0,
                                'confidence': 1.0
                            }
                            
                            # Extract word timestamps if available
                            if add_timestamps and 'result' in result:
                                words = result['result']
                                if words:
                                    segment['start'] = words[0].get('start', 0)
                                    segment['end'] = words[-1].get('end', 0)
                            
                            segments.append(segment)
                
                # Get final result
                final_result = json.loads(rec.FinalResult())
                if 'text' in final_result and final_result['text']:
                    text = final_result['text']
                    full_text += text + " "
                    
                    # Create segment for final result
                    segment = {
                        'text': text,
                        'start': 0 if not segments else segments[-1]['end'],
                        'end': 0 if not segments else segments[-1]['end'] + 5,  # Arbitrary duration
                        'confidence': 1.0
                    }
                    
                    # Extract word timestamps if available
                    if add_timestamps and 'result' in final_result:
                        words = final_result['result']
                        if words:
                            segment['start'] = words[0].get('start', segment['start'])
                            segment['end'] = words[-1].get('end', segment['end'])
                    
                    segments.append(segment)
            
            return {
                'text': full_text.strip(),
                'segments': segments
            }
            
        except Exception as e:
            logger.error(f"VOSK transcription error: {str(e)}")
            return {'text': '', 'segments': []}


# For testing the module directly
if __name__ == "__main__":
    stt_engine = SpeechToTextEngine()
    
    # Test transcription with a sample file
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        engine = sys.argv[2] if len(sys.argv) > 2 else 'whisper'
        
        params = {
            'audio_path': audio_path,
            'language': '',  # Auto-detect
            'add_timestamps': True,
            'add_punctuation': True,
            'engine': engine
        }
        
        print(f"Transcribing {audio_path} with {engine} engine...")
        result = stt_engine.transcribe(params)
        
        print(f"Transcription result:")
        print(f"Full text: {result['text']}")
        print(f"Segments: {len(result['segments'])}")
        for i, segment in enumerate(result['segments']):
            print(f"  {i+1}. [{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
    else:
        print("Usage: python stt_module.py <audio_file> [engine]")
        print("Available engines:", end=" ")
        available = []
        if stt_engine.whisper_service:
            available.append("whisper")
        if stt_engine.google_service:
            available.append("google")
        if stt_engine.azure_service:
            available.append("azure")
        if stt_engine.vosk_service:
            available.append("vosk")
        print(", ".join(available)) 