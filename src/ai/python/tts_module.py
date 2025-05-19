#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlouEdit Text-to-Speech Module
Provides functionality for converting text to speech using various TTS engines.
"""

import os
import sys
import logging
import tempfile
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tts_module")

# Try importing required dependencies
try:
    import numpy as np
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("scipy package not found. Some audio processing may be limited.")
    SCIPY_AVAILABLE = False

# Try importing TTS engines
try:
    import gtts
    GTTS_AVAILABLE = True
except ImportError:
    logger.warning("gtts package not found. Google TTS will be unavailable.")
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    logger.warning("pyttsx3 package not found. Offline TTS will be unavailable.")
    PYTTSX3_AVAILABLE = False

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_TTS_AVAILABLE = os.environ.get("AZURE_SPEECH_KEY") is not None
except ImportError:
    logger.warning("azure.cognitiveservices.speech package not found. Azure TTS will be unavailable.")
    AZURE_TTS_AVAILABLE = False

class TextToSpeechEngine:
    """Engine for converting text to speech using various backends."""
    
    def __init__(self):
        """Initialize the TTS engine."""
        self.gtts_service = GTTS_AVAILABLE
        self.pyttsx3_service = PYTTSX3_AVAILABLE
        self.azure_service = AZURE_TTS_AVAILABLE
        
        # Initialize offline engine if available
        if self.pyttsx3_service:
            self.offline_engine = pyttsx3.init()
        else:
            self.offline_engine = None
        
        # Load cached list of voices
        self._available_voices = self._load_default_voices()
    
    def get_available_voices(self, language_code=None):
        """
        Get a list of available voices.
        
        Args:
            language_code (str): Optional filter by language code (e.g., 'en-US', 'ko-KR')
            
        Returns:
            list: List of voice dictionaries with id, name, gender, and language code
        """
        voices = self._available_voices
        
        # Filter by language code if provided
        if language_code:
            voices = [v for v in voices if v['language_code'].startswith(language_code)]
        
        return voices
    
    def synthesize_speech(self, params, output_path):
        """
        Synthesize speech from text.
        
        Args:
            params (dict): Dictionary containing synthesis parameters
                - text (str): Text to synthesize
                - voice_id (str): Voice ID to use
                - speaking_rate (float): Speaking rate multiplier
                - pitch (float): Pitch adjustment
                - volume_gain_db (float): Volume gain in dB
                - add_timestamps (bool): Whether to add word timestamps
            output_path (str): Path where the generated audio should be saved
            
        Returns:
            bool: True if synthesis was successful, False otherwise
        """
        try:
            text = params.get('text', '')
            voice_id = params.get('voice_id', '')
            
            if not text:
                logger.error("No text provided for synthesis")
                return False
            
            # Determine voice service and ID parsing
            service_type, voice_params = self._parse_voice_id(voice_id)
            
            if service_type == 'gtts':
                return self._synthesize_with_gtts(text, voice_params, output_path, params)
            elif service_type == 'pyttsx3':
                return self._synthesize_with_pyttsx3(text, voice_params, output_path, params)
            elif service_type == 'azure':
                return self._synthesize_with_azure(text, voice_params, output_path, params)
            else:
                # Default to Google TTS if available
                if self.gtts_service:
                    return self._synthesize_with_gtts(text, {'lang': 'en'}, output_path, params)
                # Fall back to offline TTS
                elif self.pyttsx3_service:
                    return self._synthesize_with_pyttsx3(text, {}, output_path, params)
                else:
                    logger.error("No TTS service available")
                    return False
        
        except Exception as e:
            logger.error(f"Failed to synthesize speech: {str(e)}")
            return False
    
    def _parse_voice_id(self, voice_id):
        """Parse voice ID to determine service and parameters."""
        if not voice_id:
            # Default to Google TTS if available, otherwise offline
            if self.gtts_service:
                return 'gtts', {'lang': 'en'}
            elif self.pyttsx3_service:
                return 'pyttsx3', {}
        
        # Parse service prefix
        parts = voice_id.split('://', 1)
        if len(parts) == 2:
            service, params_str = parts
            # Parse parameters
            params = {}
            for param in params_str.split('&'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    params[key] = value
                else:
                    params['id'] = param
            
            return service, params
        
        # If no explicit service, check format to guess
        if voice_id.startswith('azure-'):
            # Azure voice name
            return 'azure', {'name': voice_id[6:]}
        elif len(voice_id) == 2 or len(voice_id) == 5:
            # Likely a language code like 'en' or 'en-US' for Google TTS
            return 'gtts', {'lang': voice_id}
        elif voice_id.isdigit():
            # Likely a voice ID for pyttsx3
            return 'pyttsx3', {'id': int(voice_id)}
        
        # Default to Google TTS with the voice_id as language
        return 'gtts', {'lang': voice_id[:2]}
    
    def _synthesize_with_gtts(self, text, voice_params, output_path, params):
        """Synthesize speech using Google TTS."""
        if not self.gtts_service:
            logger.error("Google TTS service not available")
            return False
        
        try:
            # Parse parameters
            lang = voice_params.get('lang', 'en')
            tld = voice_params.get('tld', 'com')
            slow = voice_params.get('slow', 'false').lower() == 'true'
            
            # Apply speed adjustment
            speaking_rate = float(params.get('speaking_rate', 1.0))
            if speaking_rate < 0.8:
                slow = True
            
            # Create TTS object
            tts = gtts.gTTS(text=text, lang=lang, slow=slow, tld=tld)
            
            # Save to output file
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            tts.save(output_path)
            
            # Process with post-processing if needed
            if (abs(float(params.get('pitch', 0.0))) > 0.1 or 
                abs(speaking_rate - 1.0) > 0.1 or 
                abs(float(params.get('volume_gain_db', 0.0))) > 0.1):
                self._post_process_audio(output_path, params)
            
            return True
            
        except Exception as e:
            logger.error(f"Google TTS synthesis error: {str(e)}")
            return False
    
    def _synthesize_with_pyttsx3(self, text, voice_params, output_path, params):
        """Synthesize speech using offline pyttsx3."""
        if not self.pyttsx3_service or not self.offline_engine:
            logger.error("Offline TTS service not available")
            return False
        
        try:
            # Set voice if specified
            if 'id' in voice_params:
                self.offline_engine.setProperty('voice', int(voice_params['id']))
            
            # Set speaking rate
            speaking_rate = float(params.get('speaking_rate', 1.0))
            rate = int(self.offline_engine.getProperty('rate') * speaking_rate)
            self.offline_engine.setProperty('rate', rate)
            
            # Set volume
            volume_gain_db = float(params.get('volume_gain_db', 0.0))
            volume = min(1.0, max(0.0, 1.0 + (volume_gain_db / 20.0)))  # Convert dB to multiplier
            self.offline_engine.setProperty('volume', volume)
            
            # Save to output file
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            self.offline_engine.save_to_file(text, output_path)
            self.offline_engine.runAndWait()
            
            # Process pitch using post-processing
            pitch = float(params.get('pitch', 0.0))
            if abs(pitch) > 0.1:
                self._post_process_audio(output_path, {'pitch': pitch})
            
            return True
            
        except Exception as e:
            logger.error(f"Offline TTS synthesis error: {str(e)}")
            return False
    
    def _synthesize_with_azure(self, text, voice_params, output_path, params):
        """Synthesize speech using Azure TTS."""
        if not self.azure_service:
            logger.error("Azure TTS service not available")
            return False
        
        try:
            # Get Azure credentials
            subscription_key = os.environ.get("AZURE_SPEECH_KEY")
            region = os.environ.get("AZURE_SPEECH_REGION", "eastus")
            
            # Create speech config
            speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
            
            # Set voice name
            voice_name = voice_params.get('name', 'en-US-GuyNeural')
            speech_config.speech_synthesis_voice_name = voice_name
            
            # Set output format to WAV
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
            )
            
            # Create speech synthesizer
            audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config, 
                audio_config=audio_config
            )
            
            # Build SSML with parameters
            speaking_rate = float(params.get('speaking_rate', 1.0))
            pitch = float(params.get('pitch', 0.0))
            volume = float(params.get('volume_gain_db', 0.0))
            
            # Convert pitch to SSML format
            pitch_str = f"{pitch:+.1f}st"
            
            # Convert volume to SSML format (dB to percentage)
            volume_str = f"{min(100, max(0, 100 + volume * 4)):+.0f}%"
            
            # Create SSML
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="{voice_name}">
                    <prosody rate="{speaking_rate:.1f}" pitch="{pitch_str}" volume="{volume_str}">
                        {text}
                    </prosody>
                </voice>
            </speak>
            """
            
            # Synthesize speech with SSML
            result = synthesizer.speak_ssml(ssml)
            
            # Check result
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return True
            else:
                logger.error(f"Azure TTS failed: {result.reason}")
                return False
                
        except Exception as e:
            logger.error(f"Azure TTS synthesis error: {str(e)}")
            return False
    
    def _post_process_audio(self, audio_path, params):
        """Apply post-processing effects to audio file."""
        if not SCIPY_AVAILABLE:
            logger.warning("Cannot apply post-processing: scipy not available")
            return False
        
        try:
            # Load audio file
            sample_rate, audio_data = wavfile.read(audio_path)
            
            # Apply volume adjustment
            volume_gain_db = float(params.get('volume_gain_db', 0.0))
            if abs(volume_gain_db) > 0.1:
                gain_factor = 10 ** (volume_gain_db / 20)
                audio_data = np.clip(audio_data * gain_factor, -32768, 32767).astype(np.int16)
            
            # Apply pitch adjustment (requires more complex processing)
            # This is a placeholder - real pitch shifting would require a DSP library
            
            # Save modified audio
            wavfile.write(audio_path, sample_rate, audio_data)
            return True
            
        except Exception as e:
            logger.error(f"Audio post-processing error: {str(e)}")
            return False
    
    def _load_default_voices(self):
        """Load and return a list of default voices."""
        voices = []
        
        # Add Google TTS voices
        if self.gtts_service:
            voices.extend([
                {
                    'id': 'gtts://lang=en&tld=com',
                    'name': 'Google TTS English (US)',
                    'gender': 'NEUTRAL',
                    'model': 'STANDARD',
                    'language_code': 'en-US'
                },
                {
                    'id': 'gtts://lang=ko&tld=co.kr',
                    'name': 'Google TTS Korean',
                    'gender': 'NEUTRAL',
                    'model': 'STANDARD',
                    'language_code': 'ko-KR'
                },
                {
                    'id': 'gtts://lang=ja&tld=co.jp',
                    'name': 'Google TTS Japanese',
                    'gender': 'NEUTRAL',
                    'model': 'STANDARD',
                    'language_code': 'ja-JP'
                },
                {
                    'id': 'gtts://lang=fr&tld=fr',
                    'name': 'Google TTS French',
                    'gender': 'NEUTRAL',
                    'model': 'STANDARD',
                    'language_code': 'fr-FR'
                },
            ])
        
        # Add offline TTS voices
        if self.pyttsx3_service and self.offline_engine:
            try:
                for idx, voice in enumerate(self.offline_engine.getProperty('voices')):
                    lang_code = 'en-US'  # Default
                    # Try to extract language code from voice ID
                    if 'language' in dir(voice) and voice.language:
                        lang_code = voice.language
                    
                    gender = 'NEUTRAL'
                    if 'gender' in dir(voice) and voice.gender:
                        gender = voice.gender
                    
                    voices.append({
                        'id': f'pyttsx3://id={idx}',
                        'name': f"Local TTS: {voice.name}",
                        'gender': gender,
                        'model': 'STANDARD',
                        'language_code': lang_code
                    })
            except Exception as e:
                logger.error(f"Error loading pyttsx3 voices: {str(e)}")
        
        # Add Azure voices
        if self.azure_service:
            # Add a few common Azure voices
            voices.extend([
                {
                    'id': 'azure://name=ko-KR-SoonBokNeural',
                    'name': 'Azure Korean Female (SoonBok)',
                    'gender': 'FEMALE',
                    'model': 'NEURAL',
                    'language_code': 'ko-KR'
                },
                {
                    'id': 'azure://name=ko-KR-InJoonNeural',
                    'name': 'Azure Korean Male (InJoon)',
                    'gender': 'MALE',
                    'model': 'NEURAL',
                    'language_code': 'ko-KR'
                },
                {
                    'id': 'azure://name=en-US-JennyNeural',
                    'name': 'Azure English US Female (Jenny)',
                    'gender': 'FEMALE',
                    'model': 'NEURAL',
                    'language_code': 'en-US'
                },
                {
                    'id': 'azure://name=en-US-GuyNeural',
                    'name': 'Azure English US Male (Guy)',
                    'gender': 'MALE',
                    'model': 'NEURAL',
                    'language_code': 'en-US'
                },
            ])
        
        return voices


# For testing the module directly
if __name__ == "__main__":
    tts_engine = TextToSpeechEngine()
    
    # Get available voices
    voices = tts_engine.get_available_voices()
    print(f"Found {len(voices)} voices")
    for i, voice in enumerate(voices[:3]):  # Show first 3 only
        print(f"{i+1}. {voice['name']} ({voice['language_code']})")
    
    # Test synthesis
    if len(voices) > 0:
        params = {
            'text': 'This is a test of the text to speech system.',
            'voice_id': voices[0]['id'],
            'speaking_rate': 1.0,
            'pitch': 0.0,
            'volume_gain_db': 0.0
        }
        
        output_path = "test_speech.wav"
        success = tts_engine.synthesize_speech(params, output_path)
        
        if success:
            print(f"Speech generated successfully at {output_path}")
        else:
            print("Failed to generate speech") 