"""
WebSocket client for real-time audio transcription with Whisper.

This module provides streaming audio transcription capabilities using
WebSocket connections to the Lemonade server.

Note: Requires websocket-client package: pip install websocket-client
"""

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    websocket = None

import json
import base64
import os
from typing import Optional, Callable, Generator, List
from urllib.parse import urlencode


class WhisperWebSocketClient:
    """
    Client for real-time audio transcription via WebSocket.
    
    Connects to Lemonade's /realtime endpoint for streaming
    speech-to-text using Whisper models.
    
    Example:
        ```python
        client = WhisperWebSocketClient()
        client.connect(model="Whisper-Tiny")
        
        # Stream audio file
        for transcription in client.stream("audio.pcm"):
            print(f"Heard: {transcription}")
        
        client.disconnect()
        ```
    """
    
    def __init__(self, base_url: str = "ws://localhost:9000"):
        """
        Initialize the WebSocket client.
        
        Args:
            base_url: WebSocket base URL (default: ws://localhost:9000)
        """
        self.base_url = base_url
        self.ws: Optional[websocket.WebSocket] = None
        self._transcription_callback: Optional[Callable[[str], None]] = None
        self._message_queue: List[str] = []
        self._connected = False
        
    def connect(self, model: str = "Whisper-Tiny", timeout: int = 10) -> bool:
        """
        Establish WebSocket connection to the transcription server.
        
        Args:
            model: Whisper model to use (Whisper-Tiny, Whisper-Base, Whisper-Small)
            timeout: Connection timeout in seconds
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not WEBSOCKET_AVAILABLE:
            print("websocket-client not installed. Run: pip install websocket-client")
            return False
        
        try:
            url = f"{self.base_url}/realtime?{urlencode({'model': model})}"
            self.ws = websocket.create_connection(url, timeout=timeout)
            self._connected = True
            self._message_queue = []
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """Close the WebSocket connection."""
        if self.ws:
            try:
                self.ws.close()
            except:
                pass
        self.ws = None
        self._connected = False
    
    def is_connected(self) -> bool:
        """Check if the WebSocket connection is active."""
        return self._connected and self.ws is not None
    
    def send_audio(self, audio_data: bytes) -> bool:
        """
        Send audio chunk to the server.
        
        Args:
            audio_data: Raw PCM16 audio data (16kHz, mono)
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.is_connected():
            print("Not connected to WebSocket server")
            return False
        
        try:
            # Encode audio as base64
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            
            message = {
                "type": "input_audio_buffer.append",
                "audio": audio_b64
            }
            
            self.ws.send(json.dumps(message))
            return True
        except Exception as e:
            print(f"Error sending audio: {e}")
            return False
    
    def commit(self) -> bool:
        """
        Trigger transcription of the current audio buffer.
        
        Returns:
            bool: True if commit sent successfully, False otherwise
        """
        if not self.is_connected():
            print("Not connected to WebSocket server")
            return False
        
        try:
            message = {"type": "input_audio_buffer.commit"}
            self.ws.send(json.dumps(message))
            return True
        except Exception as e:
            print(f"Error committing buffer: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Clear the audio buffer without transcribing.
        
        Returns:
            bool: True if clear sent successfully, False otherwise
        """
        if not self.is_connected():
            print("Not connected to WebSocket server")
            return False
        
        try:
            message = {"type": "input_audio_buffer.clear"}
            self.ws.send(json.dumps(message))
            return True
        except Exception as e:
            print(f"Error clearing buffer: {e}")
            return False
    
    def configure_session(
        self,
        threshold: float = 0.5,
        prefix_padding_ms: int = 300,
        silence_duration_ms: int = 500
    ) -> bool:
        """
        Configure VAD (Voice Activity Detection) settings.
        
        Args:
            threshold: VAD threshold (0.0 - 1.0)
            prefix_padding_ms: Padding before speech starts
            silence_duration_ms: Silence duration to trigger end of speech
            
        Returns:
            bool: True if configuration sent successfully, False otherwise
        """
        if not self.is_connected():
            print("Not connected to WebSocket server")
            return False
        
        try:
            message = {
                "type": "session.update",
                "session": {
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": threshold,
                        "prefix_padding_ms": prefix_padding_ms,
                        "silence_duration_ms": silence_duration_ms
                    }
                }
            }
            self.ws.send(json.dumps(message))
            return True
        except Exception as e:
            print(f"Error configuring session: {e}")
            return False
    
    def on_transcription(self, callback: Callable[[str], None]):
        """
        Set callback function for transcription results.
        
        Args:
            callback: Function to call with transcription text
        """
        self._transcription_callback = callback
    
    def _receive_message(self, timeout: float = 1.0) -> Optional[dict]:
        """
        Receive and parse a WebSocket message.
        
        Args:
            timeout: Receive timeout in seconds
            
        Returns:
            Parsed message dict or None
        """
        if not self.is_connected():
            return None
        
        try:
            self.ws.settimeout(timeout)
            data = self.ws.recv()
            return json.loads(data)
        except websocket.WebSocketTimeoutException:
            return None
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None
    
    def _process_message(self, message: dict) -> Optional[str]:
        """
        Process a WebSocket message and extract transcription.
        
        Args:
            message: Raw message dict from server
            
        Returns:
            Transcription text or None
        """
        event_type = message.get("type", "")
        
        # Handle transcription completed
        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = message.get("transcript", "")
            if transcript and self._transcription_callback:
                self._transcription_callback(transcript)
            return transcript
        
        # Handle speech started/stopped events
        elif event_type == "input_audio_buffer.speech_started":
            print(f"Speech detected at {message.get('audio_start_ms', 0)}ms")
        
        elif event_type == "input_audio_buffer.speech_stopped":
            print(f"Speech ended at {message.get('audio_end_ms', 0)}ms")
        
        # Handle session created
        elif event_type == "session.created":
            print(f"Session established: {message.get('id', 'unknown')}")
        
        # Handle errors
        elif event_type == "error":
            print(f"WebSocket error: {message.get('message', 'Unknown error')}")
        
        return None
    
    def stream(self, audio_file: str, chunk_size: int = 32000) -> Generator[str, None, None]:
        """
        Stream an audio file and yield transcriptions.
        
        Args:
            audio_file: Path to PCM16 audio file (16kHz, mono)
            chunk_size: Number of samples per chunk (default: 32000 = 2 seconds)
            
        Yields:
            Transcription text chunks as they become available
            
        Example:
            ```python
            for text in client.stream("audio.pcm"):
                print(f"Heard: {text}")
            ```
        """
        if not self.is_connected():
            print("Not connected. Call connect() first.")
            return
        
        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            return
        
        try:
            with open(audio_file, "rb") as f:
                while True:
                    chunk = f.read(chunk_size * 2)  # 2 bytes per sample (PCM16)
                    if not chunk:
                        break
                    
                    # Send audio chunk
                    self.send_audio(chunk)
                    
                    # Check for transcriptions (non-blocking)
                    while True:
                        message = self._receive_message(timeout=0.1)
                        if message is None:
                            break
                        
                        transcription = self._process_message(message)
                        if transcription:
                            yield transcription
            
            # Commit final buffer
            self.commit()
            
            # Collect remaining transcriptions
            while True:
                message = self._receive_message(timeout=1.0)
                if message is None:
                    break
                
                transcription = self._process_message(message)
                if transcription:
                    yield transcription
                    
        except Exception as e:
            print(f"Error streaming audio: {e}")
    
    def stream_microphone(
        self,
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        channels: int = 1
    ) -> Generator[str, None, None]:
        """
        Stream from microphone and yield transcriptions.
        
        Note: Requires pyaudio package: pip install pyaudio
        
        Args:
            device_index: Microphone device index (None = default)
            sample_rate: Sample rate in Hz (default: 16000)
            channels: Number of channels (default: 1 for mono)
            
        Yields:
            Transcription text chunks
        """
        try:
            import pyaudio
        except ImportError:
            print("pyaudio not installed. Run: pip install pyaudio")
            return
        
        if not self.is_connected():
            print("Not connected. Call connect() first.")
            return
        
        try:
            audio = pyaudio.PyAudio()
            
            # Get device info
            if device_index is None:
                device_index = audio.get_default_input_device_info()["index"]
            
            # Open stream
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024
            )
            
            print("Listening... (press Ctrl+C to stop)")
            
            try:
                while True:
                    # Read audio data
                    data = stream.read(1024, exception_on_overflow=False)
                    
                    # Send to server
                    self.send_audio(data)
                    
                    # Check for transcriptions
                    message = self._receive_message(timeout=0.1)
                    if message:
                        transcription = self._process_message(message)
                        if transcription:
                            yield transcription
                            
            except KeyboardInterrupt:
                print("\nStopping...")
            finally:
                stream.stop_stream()
                stream.close()
                audio.terminate()
                
        except Exception as e:
            print(f"Error streaming from microphone: {e}")
