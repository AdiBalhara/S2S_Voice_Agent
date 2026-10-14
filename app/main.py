"""
Main application module.

Provides the FastAPI server with WebSocket endpoint for Gemini Live
bidirectional audio streaming, plus the ADK web UI integration.
"""

import asyncio
import base64
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

from app.database.connection import configure_windows_event_loop

# On Windows, psycopg async requires SelectorEventLoop instead of ProactorEventLoop
configure_windows_event_loop()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from google.adk import Runner
from google.adk.agents import LiveRequestQueue
from google.adk.runners import RunConfig
from google.genai import types as genai_types

from app.agent.agent import create_agent
from app.config import (
    GEMINI_API_KEY,
    GEMINI_LIVE_MODEL,
    SERVER_HOST,
    SERVER_PORT,
    log_config_summary,
    validate_config,
)
from app.persistence.sessions import get_session_service

logger = logging.getLogger(__name__)

# Application name constant
APP_NAME = "trijotech-voice-agent"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting TrijoTech Voice Agent...")

    errors = validate_config()
    if errors:
        for err in errors:
            logger.error("Configuration error: %s", err)
        logger.error("Fix the above errors before starting.")
    else:
        log_config_summary()
        logger.info("Configuration validated successfully.")

    # Set API key
    if GEMINI_API_KEY:
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

    yield

    # Shutdown
    logger.info("Shutting down TrijoTech Voice Agent...")


app = FastAPI(
    title="TrijoTech Voice Sales Agent",
    description="Real-time voice sales agent powered by Google ADK + Gemini Live",
    version="0.1.0",
    lifespan=lifespan,
)


# HTML page for the voice agent interface
VOICE_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrijoTech Voice Agent</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0d0d2b 100%);
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .container {
            text-align: center;
            padding: 2rem;
            max-width: 600px;
        }
        h1 {
            font-size: 2rem;
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            color: #8888aa;
            margin-bottom: 2rem;
            font-size: 0.95rem;
        }
        #status {
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 2rem;
            font-size: 0.9rem;
        }
        .btn {
            padding: 1rem 2.5rem;
            border: none;
            border-radius: 50px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 0.5rem;
        }
        .btn-start {
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            color: white;
            box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
        }
        .btn-start:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 30px rgba(59, 130, 246, 0.5);
        }
        .btn-stop {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
            box-shadow: 0 4px 20px rgba(239, 68, 68, 0.3);
        }
        .btn-stop:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 30px rgba(239, 68, 68, 0.5);
        }
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }
        .pulse {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            vertical-align: middle;
        }
        .pulse-active {
            background: #22c55e;
            animation: pulse 1.5s infinite;
        }
        .pulse-inactive {
            background: #6b7280;
        }
        .pulse-connecting {
            background: #f59e0b;
            animation: pulse 0.8s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.2); }
        }
        .transcript {
            margin-top: 2rem;
            text-align: left;
            max-height: 300px;
            overflow-y: auto;
            padding: 1rem;
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.05);
            font-size: 0.85rem;
            line-height: 1.6;
        }
        .transcript .user { color: #60a5fa; }
        .transcript .agent { color: #a78bfa; }
        .transcript .system { color: #6b7280; font-style: italic; }
        .info {
            margin-top: 1.5rem;
            font-size: 0.8rem;
            color: #555;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>TrijoTech Voice Agent</h1>
        <p class="subtitle">Powered by Google ADK + Gemini Live</p>
        
        <div id="status">
            <span class="pulse pulse-inactive" id="statusDot"></span>
            <span id="statusText">Ready to connect</span>
        </div>

        <div>
            <button class="btn btn-start" id="startBtn" onclick="startConversation()">
                🎤 Start Conversation
            </button>
            <button class="btn btn-stop" id="stopBtn" onclick="stopConversation()" disabled>
                ⏹ End Conversation
            </button>
        </div>

        <div class="transcript" id="transcript"></div>
        
        <p class="info">
            Ensure microphone permissions are granted. 
            Speak naturally — Trija will respond via voice.
        </p>
    </div>

    <script>
        let ws = null;
        let mediaStream = null;
        let audioContext = null;
        let processor = null;
        let playbackContext = null;
        let nextStartTime = 0;
        let activeSources = [];
        let currentAgentP = null;
        let currentUserP = null;

        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const transcript = document.getElementById('transcript');

        function setStatus(state, text) {
            statusDot.className = 'pulse pulse-' + state;
            statusText.textContent = text;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function addTranscript(role, text) {
            currentAgentP = null;
            currentUserP = null;
            const p = document.createElement('p');
            p.className = role;
            const label = role === 'user' ? '🧑 You' : role === 'agent' ? '🤖 Trija' : '⚙️ System';
            p.innerHTML = '<strong>' + label + ':</strong> ' + escapeHtml(text);
            transcript.appendChild(p);
            transcript.scrollTop = transcript.scrollHeight;
            return p;
        }

        function appendAgentTranscript(text) {
            currentUserP = null;
            if (!currentAgentP) {
                currentAgentP = document.createElement('p');
                currentAgentP.className = 'agent';
                currentAgentP.innerHTML = '<strong>🤖 Trija:</strong> <span></span>';
                transcript.appendChild(currentAgentP);
            }
            const span = currentAgentP.querySelector('span');
            span.textContent += text;
            transcript.scrollTop = transcript.scrollHeight;
        }

        function finalizeAgentTranscript(text) {
            if (currentAgentP && text) {
                const span = currentAgentP.querySelector('span');
                if (text.length >= span.textContent.length) {
                    span.textContent = text;
                }
            } else if (text) {
                appendAgentTranscript(text);
            }
            currentAgentP = null;
            transcript.scrollTop = transcript.scrollHeight;
        }

        function appendUserTranscript(text) {
            currentAgentP = null;
            if (!currentUserP) {
                currentUserP = document.createElement('p');
                currentUserP.className = 'user';
                currentUserP.innerHTML = '<strong>🧑 You:</strong> <span></span>';
                transcript.appendChild(currentUserP);
            }
            const span = currentUserP.querySelector('span');
            span.textContent = text;
            transcript.scrollTop = transcript.scrollHeight;
        }

        async function startConversation() {
            try {
                setStatus('connecting', 'Connecting...');
                startBtn.disabled = true;

                currentAgentP = null;
                currentUserP = null;
                nextStartTime = 0;

                // Get microphone
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                    }
                });

                // Setup audio processing
                audioContext = new AudioContext({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(mediaStream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);

                // Mute gain node to prevent mic loopback into local speakers
                const muteNode = audioContext.createGain();
                muteNode.gain.value = 0;
                
                // Connect to WebSocket
                const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(protocol + '//' + location.host + '/ws/audio');

                ws.onopen = () => {
                    setStatus('active', 'Connected — speak now');
                    stopBtn.disabled = false;
                    addTranscript('system', 'Connected to voice agent');

                    // Start sending audio without playing mic into speakers
                    source.connect(processor);
                    processor.connect(muteNode);
                    muteNode.connect(audioContext.destination);

                    processor.onaudioprocess = (e) => {
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            const float32 = e.inputBuffer.getChannelData(0);
                            const int16 = new Int16Array(float32.length);
                            for (let i = 0; i < float32.length; i++) {
                                int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
                            }
                            const b64 = btoa(String.fromCharCode(...new Uint8Array(int16.buffer)));
                            ws.send(JSON.stringify({ type: 'audio', data: b64 }));
                        }
                    };
                };

                ws.onmessage = (event) => {
                    const msg = JSON.parse(event.data);
                    if (msg.type === 'audio') {
                        playAudio(msg.data);
                    } else if (msg.type === 'transcript_agent') {
                        appendAgentTranscript(msg.text);
                    } else if (msg.type === 'transcript_agent_final') {
                        finalizeAgentTranscript(msg.text);
                    } else if (msg.type === 'transcript_user') {
                        appendUserTranscript(msg.text);
                    } else if (msg.type === 'tool_call') {
                        addTranscript('system', 'Using tool: ' + msg.tool);
                    } else if (msg.type === 'error') {
                        addTranscript('system', 'Error: ' + msg.message);
                    }
                };

                ws.onclose = () => {
                    setStatus('inactive', 'Disconnected');
                    addTranscript('system', 'Connection closed');
                    cleanup();
                };

                ws.onerror = (err) => {
                    setStatus('inactive', 'Connection error');
                    addTranscript('system', 'Connection error occurred');
                    cleanup();
                };

            } catch (err) {
                setStatus('inactive', 'Failed to start');
                addTranscript('system', 'Error: ' + err.message);
                startBtn.disabled = false;
            }
        }

        function playAudio(base64Data) {
            if (!playbackContext) {
                playbackContext = new AudioContext({ sampleRate: 24000 });
            }
            if (playbackContext.state === 'suspended') {
                playbackContext.resume();
            }

            const binaryStr = atob(base64Data);
            const bytes = new Uint8Array(binaryStr.length);
            for (let i = 0; i < binaryStr.length; i++) {
                bytes[i] = binaryStr.charCodeAt(i);
            }
            const int16 = new Int16Array(bytes.buffer);
            const float32 = new Float32Array(int16.length);
            for (let i = 0; i < int16.length; i++) {
                float32[i] = int16[i] / 32768.0;
            }
            const buffer = playbackContext.createBuffer(1, float32.length, 24000);
            buffer.getChannelData(0).set(float32);
            const source = playbackContext.createBufferSource();
            source.buffer = buffer;
            source.connect(playbackContext.destination);

            // Sequential timeline scheduling
            const now = playbackContext.currentTime;
            if (nextStartTime < now) {
                nextStartTime = now;
            }
            source.start(nextStartTime);
            nextStartTime += buffer.duration;

            activeSources.push(source);
            source.onended = () => {
                const idx = activeSources.indexOf(source);
                if (idx !== -1) {
                    activeSources.splice(idx, 1);
                }
            };
        }

        function stopAllAudio() {
            for (const s of activeSources) {
                try { s.stop(); } catch (e) {}
            }
            activeSources = [];
            if (playbackContext) {
                nextStartTime = playbackContext.currentTime;
            } else {
                nextStartTime = 0;
            }
        }

        function stopConversation() {
            if (ws) {
                ws.send(JSON.stringify({ type: 'stop' }));
                ws.close();
            }
            cleanup();
        }

        function cleanup() {
            stopAllAudio();
            if (processor) { processor.disconnect(); processor = null; }
            if (audioContext) { audioContext.close(); audioContext = null; }
            if (playbackContext) { playbackContext.close(); playbackContext = null; }
            if (mediaStream) {
                mediaStream.getTracks().forEach(t => t.stop());
                mediaStream = null;
            }
            ws = null;
            nextStartTime = 0;
            currentAgentP = null;
            currentUserP = null;
            startBtn.disabled = false;
            stopBtn.disabled = true;
            setStatus('inactive', 'Ready to connect');
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the voice agent web interface."""
    return VOICE_UI_HTML


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "agent": "trijotech-voice-agent"}


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for bidirectional audio streaming with Gemini Live.
    
    Receives audio chunks from the browser microphone, feeds them to the
    ADK agent via run_live, and sends back audio responses.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted.")

    session_id = str(uuid.uuid4())
    user_id = "web_user"

    try:
        # Create agent and runner
        agent = create_agent()
        session_service = get_session_service()

        # Create a new session
        session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
        )
        session_id = session.id
        logger.info("Session created: %s", session_id)

        runner = Runner(
            agent=agent,
            app_name=APP_NAME,
            session_service=session_service,
        )

        # Create a LiveRequestQueue for feeding audio to the agent
        live_request_queue = LiveRequestQueue()

        # Configure the live run
        run_config = RunConfig(
            response_modalities=[genai_types.Modality.AUDIO],
        )

        # Start the live agent loop in background
        live_events_task = asyncio.create_task(
            _process_live_events(runner, session, live_request_queue, run_config, websocket)
        )

        # Receive audio from client and feed to live request queue
        try:
            while True:
                raw_message = await websocket.receive_text()
                msg = json.loads(raw_message)

                if msg.get("type") == "audio":
                    audio_bytes = base64.b64decode(msg["data"])
                    live_request_queue.send_realtime(
                        genai_types.Blob(
                            mime_type=msg.get("mime_type", "audio/pcm;rate=16000"),
                            data=audio_bytes,
                        )
                    )

                elif msg.get("type") == "stop":
                    logger.info("Client requested stop.")
                    live_request_queue.send_audio_stream_end()
                    break

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected by client.")

        finally:
            # Signal the live request queue to close
            live_request_queue.close()
            # Wait for live events task to finish
            if not live_events_task.done():
                live_events_task.cancel()
                try:
                    await live_events_task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

    finally:
        logger.info("WebSocket session %s ended.", session_id)


try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(f):
            return f
        return decorator


@traceable(name="gemini_live_session", run_type="chain")
async def _process_live_events(runner, session, live_request_queue, run_config, websocket):
    """Process live events from the ADK runner and send responses to the client."""
    try:
        async for event in runner.run_live(
            session=session,
            live_request_queue=live_request_queue,
            run_config=run_config,
        ):
            # Handle ADK Event content (audio and text)
            if hasattr(event, "content") and event.content and hasattr(event.content, "parts"):
                for part in event.content.parts:
                    # Audio data from model
                    if hasattr(part, "inline_data") and part.inline_data:
                        audio_b64 = base64.b64encode(
                            part.inline_data.data
                        ).decode("utf-8")
                        await websocket.send_json({
                            "type": "audio",
                            "data": audio_b64,
                        })

                    # Text transcript from model
                    if hasattr(part, "text") and part.text:
                        await websocket.send_json({
                            "type": "transcript_agent",
                            "text": part.text,
                        })

            # Handle live model output transcription if not already in content parts
            elif hasattr(event, "output_transcription") and event.output_transcription:
                text = getattr(event.output_transcription, "text", None)
                if text:
                    await websocket.send_json({
                        "type": "transcript_agent_final",
                        "text": text,
                    })

            # Handle user input transcription
            if hasattr(event, "input_transcription") and event.input_transcription:
                text = getattr(event.input_transcription, "text", None)
                if text:
                    await websocket.send_json({
                        "type": "transcript_user",
                        "text": text,
                    })

            # Handle tool calls
            if hasattr(event, "get_function_calls") and callable(event.get_function_calls):
                for fc in event.get_function_calls():
                    await websocket.send_json({
                        "type": "tool_call",
                        "tool": getattr(fc, "name", "unknown"),
                    })

    except asyncio.CancelledError:
        logger.info("Live events processing cancelled.")
    except Exception as e:
        logger.error("Error in live events processing: %s", e, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


def run_server():
    """Run the FastAPI server."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting server on %s:%d", SERVER_HOST, SERVER_PORT)
    uvicorn.run(
        "app.main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
    )


if __name__ == "__main__":
    run_server()
