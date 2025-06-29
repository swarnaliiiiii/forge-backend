#!/usr/bin/env python3
"""
Screenshot Analysis Backend System with ElevenLabs Voice AI Integration
Handles Gemini API integration, Supabase storage, and ElevenLabs voice AI
API keys are handled server-side only for security
"""

import os
import json
import base64
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile
import dotenv
import threading
import queue
import signal
import time

import aiohttp
import asyncpg
from supabase import create_client, Client
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import google.generativeai as genai
from PIL import Image
import io
import traceback
import logging

# ElevenLabs imports
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, ConversationInitiationData
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables first
dotenv.load_dotenv()

# Configuration - All keys are server-side only
class Config:
    def __init__(self):
        # Try multiple .env file locations
        env_paths = [
            ".env",
            "temp/backend/.env",
            "../.env",
            "../../.env"
        ]
        
        env_loaded = False
        for env_path in env_paths:
            if os.path.exists(env_path):
                dotenv.load_dotenv(dotenv_path=env_path)
                print(f"Loaded environment from: {env_path}")
                env_loaded = True
                break
        
        if not env_loaded:
            print("Warning: No .env file found. Make sure environment variables are set.")
        
        # Get environment variables
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        self.SUPABASE_URL = os.getenv("SUPABASE_URL")
        self.SUPABASE_KEY = os.getenv("SUPABASE_KEY")
        self.ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
        self.AGENT_ID = os.getenv("AGENT_ID")
        
        # Debug: Print what we found (without revealing the actual keys)
        print(f"GEMINI_API_KEY found: {'Yes' if self.GEMINI_API_KEY else 'No'}")
        print(f"SUPABASE_URL found: {'Yes' if self.SUPABASE_URL else 'No'}")
        print(f"SUPABASE_KEY found: {'Yes' if self.SUPABASE_KEY else 'No'}")
        print(f"ELEVENLABS_API_KEY found: {'Yes' if self.ELEVENLABS_API_KEY else 'No'}")
        print(f"AGENT_ID found: {'Yes' if self.AGENT_ID else 'No'}")
        
        # Create temp directory
        self.TEMP_DIR = Path(tempfile.gettempdir()) / "screenshot_analysis"
        self.TEMP_DIR.mkdir(exist_ok=True)
        
        # Validate required environment variables
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required. Please check your .env file.")
        if not self.SUPABASE_URL:
            raise ValueError("SUPABASE_URL environment variable is required. Please check your .env file.")
        if not self.SUPABASE_KEY:
            raise ValueError("SUPABASE_KEY environment variable is required. Please check your .env file.")
        if not self.ELEVENLABS_API_KEY:
            raise ValueError("ELEVENLABS_API_KEY environment variable is required. Please check your .env file.")
        if not self.AGENT_ID:
            raise ValueError("AGENT_ID environment variable is required. Please check your .env file.")

config = Config()

# Pydantic models
class ScreenshotData(BaseModel):
    filename: str
    image_data: str  # base64 encoded
    timestamp: str

class AnalysisRequest(BaseModel):
    screenshot: ScreenshotData
    user_id: Optional[str] = None
    voice_triggered: Optional[bool] = False

class AnalysisResult(BaseModel):
    analysis_data: Dict[Any, Any]
    timestamp: str
    success: bool
    error_message: Optional[str] = None

# Gemini prompt (same as before)
GEMINI_PROMPT = """Your task is to act as an expert UI/UX analyst specializing in application interface identification and modern web-based development tools.

Analyze the provided application screenshot thoroughly. Your goal is to generate a comprehensive UI analysis in a structured JSON format. This JSON will be used by another agent that requires full textual context of the screenshot's UI.

You MUST return a single JSON object. Do not include any introductory or concluding text, explanations, or any formatting outside of the JSON object itself. Ensure the JSON is valid and complete.

JSON Structure Requirements:

{
"application_identification": {
"application_name": "string (specific name, e.g., 'Visual Studio Code', 'Bolt.new', 'Adobe Photoshop'. Infer if not explicitly visible but highly confident.)",
"application_type": "string (one of: 'creative', 'coding', 'other'. Refer to categories below.)",
"confidence_score": "float (between 0.0 and 1.0, representing certainty of identification. Refer to guidelines below.)",
"is_browser_based": "boolean (true if running within a web browser, false otherwise.)"
},
"interface_analysis": {
"primary_ui_elements": "array of strings (common, top-level UI components visible, e.g., 'toolbar', 'sidebar', 'main_canvas', 'status_bar', 'address_bar', 'tabs', 'menu_bar', 'scroll_bar', 'popup_dialog')",
"distinctive_features": "array of strings (unique or highly recognizable visual elements that aided identification, e.g., 'specific logo', 'unique button layout', 'branded color scheme', 'specific icon set', 'unique text editor font', 'terminal output')",
"branding_indicators": "array of strings (elements explicitly showing brand, e.g., 'logos', 'specific color_schemes', 'unique typography', 'application_name_in_title_bar', 'website_domain_in_url')"
},
"user_context": {
"current_activity": "string (brief description of what the user appears to be doing or trying to accomplish based on the UI state, e.g., 'editing code', 'designing a UI layout', 'configuring settings', 'browsing files', 'debugging an application')",
"workflow_stage": "string (one of: 'beginning', 'working', 'reviewing', 'debugging', 'configuring', 'idle', 'deploying', 'exploring')",
"content_type": "string (type of content predominantly displayed, e.g., 'code', 'design', 'document', 'settings', 'data', 'terminal_output', 'image_editing', 'video_editing', 'audio_editing', 'system_information', 'web_page')"
},
"technical_classification": {
"platform_type": "string (one of: 'desktop_native', 'web_application', 'hybrid'. 'Hybrid' for Electron-like apps that are not obviously browser-hosted.)",
"detected_technologies": "array of strings (inferred underlying technologies or integrations, e.g., 'React', 'Python', 'Node.js', 'Kubernetes', 'AI_services', 'design_tools', 'cloud_deployment', 'GitHub_integration', 'voice_AI')",
"development_environment": "boolean (true if it's an IDE, code editor, source control client, or development platform; false otherwise.)"
},
"identification_reasoning": "string (detailed explanation of the identification process, citing specific visual evidence and logical deductions that led to the conclusions in the other fields. This should be comprehensive as it provides the 'why' for the other agent.)"
}"""

# FastAPI app
app = FastAPI(title="Voice AI + Screenshot Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for voice AI
voice_conversation = None
voice_thread = None
voice_active = False
transcript_queue = queue.Queue()
screenshot_trigger_queue = queue.Queue()

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {str(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {str(exc)}",
            "type": type(exc).__name__,
            "traceback": traceback.format_exc().split('\n')[-10:]
        }
    )

class VoiceAIManager:
    def __init__(self):
        self.elevenlabs = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
        self.conversation = None
        self.is_active = False
    
    def user_transcript_callback(self, transcript):
        """Callback for user speech"""
        logger.info(f"User said: {transcript}")
        transcript_data = {
            "type": "transcript",
            "text": transcript,
            "isUser": True,
            "timestamp": datetime.now().isoformat()
        }
        transcript_queue.put(transcript_data)
        
        # Check for screenshot trigger
        if "take a screenshot" in transcript.lower():
            logger.info("Screenshot trigger detected from voice!")
            screenshot_trigger_queue.put({"voice_triggered": True})
    
    def agent_response_callback(self, response):
        """Callback for agent response"""
        logger.info(f"Agent responded: {response}")
        transcript_data = {
            "type": "transcript",
            "text": response,
            "isUser": False,
            "timestamp": datetime.now().isoformat()
        }
        transcript_queue.put(transcript_data)
    
    def start_voice_ai(self):
        """Start the ElevenLabs voice AI conversation"""
        try:
            logger.info("Starting ElevenLabs voice AI...")
            
            dynamic_vars = {
                "user_name": "User",
                "user_id": "user1",
            }

            config_data = ConversationInitiationData(
                dynamic_variables=dynamic_vars
            )

            self.conversation = Conversation(
                self.elevenlabs,
                config.AGENT_ID,
                config=config_data,
                requires_auth=bool(config.ELEVENLABS_API_KEY),
                audio_interface=DefaultAudioInterface(),
                callback_agent_response=self.agent_response_callback,
                callback_agent_response_correction=lambda original, corrected: logger.info(f"Agent correction: {original} -> {corrected}"),
                callback_user_transcript=self.user_transcript_callback,
            )

            self.conversation.start_session()
            self.is_active = True
            logger.info("ElevenLabs voice AI started successfully")
            
            # Keep the conversation alive
            while self.is_active:
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Voice AI error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.is_active = False
    
    def stop_voice_ai(self):
        """Stop the voice AI conversation"""
        try:
            self.is_active = False
            if self.conversation:
                self.conversation.end_session()
                self.conversation = None
            logger.info("Voice AI stopped")
        except Exception as e:
            logger.error(f"Error stopping voice AI: {str(e)}")

class ScreenshotAnalyzer:
    def __init__(self):
        self.temp_files = []
        try:
            logger.info("Initializing Gemini client...")
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini client initialized successfully")
            
            logger.info("Initializing Supabase client...")
            self.supabase: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            logger.info("Supabase client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize clients: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def save_temp_image(self, base64_data: str, filename: str) -> Path:
        """Save base64 image data to temporary file"""
        try:
            logger.info(f"Saving temp image: {filename}")
            
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            
            image_data = base64.b64decode(base64_data)
            logger.info(f"Decoded image data size: {len(image_data)} bytes")
            
            temp_path = config.TEMP_DIR / f"{datetime.now().timestamp()}_{filename}"
            
            with open(temp_path, 'wb') as f:
                f.write(image_data)
            
            self.temp_files.append(temp_path)
            logger.info(f"Temp image saved to: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=400, detail=f"Failed to save image: {str(e)}")
    
    def cleanup_temp_files(self):
        """Remove all temporary files"""
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                print(f"Failed to delete temp file {temp_file}: {e}")
        self.temp_files.clear()
    
    async def analyze_with_gemini(self, image_path: Path) -> Dict[Any, Any]:
        """Analyze screenshot using Gemini Vision API"""
        try:
            logger.info(f"Starting Gemini analysis for: {image_path}")
            
            image = Image.open(image_path)
            logger.info(f"Image loaded successfully. Size: {image.size}, Mode: {image.mode}")
            
            logger.info("Sending request to Gemini...")
            response = self.gemini_model.generate_content([GEMINI_PROMPT, image])
            logger.info("Received response from Gemini")
            
            analysis_text = response.text.strip()
            logger.info(f"Response text length: {len(analysis_text)}")
            logger.info(f"Response preview: {analysis_text[:200]}...")
            
            # Clean up response
            if analysis_text.startswith('```json'):
                analysis_text = analysis_text[7:-3]
            elif analysis_text.startswith('```'):
                analysis_text = analysis_text[3:-3]
            
            try:
                analysis_data = json.loads(analysis_text)
                logger.info("Successfully parsed JSON response")
                return analysis_data
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON parsing failed: {str(json_err)}")
                logger.error(f"Raw response text: {analysis_text}")
                return {
                    "error": "Failed to parse Gemini response",
                    "raw_response": analysis_text[:500],
                    "json_error": str(json_err)
                }
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Gemini analysis failed: {str(e)}")
    
    async def save_to_supabase(self, analysis_data: Dict[Any, Any], timestamp: str, user_id: str = None, voice_triggered: bool = False) -> bool:
        """Save analysis results to Supabase"""
        try:
            logger.info("Saving analysis to Supabase...")
            
            app_identification = analysis_data.get("application_identification", {})
            
            insert_data = {
                "user_id": user_id,
                "timestamp": timestamp,
                "application_type": app_identification.get("application_type"),
                "application_name": app_identification.get("application_name"),
                "analysis_data": analysis_data,
            }
            
            insert_data = {k: v for k, v in insert_data.items() if v is not None}
            
            result = self.supabase.table("screen_analyses").insert(insert_data).execute()
            logger.info("Successfully saved to Supabase")
            
            return True
            
        except Exception as e:
            logger.error(f"Supabase save failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

# Initialize components
try:
    logger.info("Initializing components...")
    analyzer = ScreenshotAnalyzer()
    voice_manager = VoiceAIManager()
    logger.info("All components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    analyzer = None
    voice_manager = None

# Voice AI endpoints
@app.post("/start-voice")
async def start_voice():
    """Start the ElevenLabs voice AI"""
    global voice_thread, voice_active
    
    try:
        if voice_active:
            return {"message": "Voice AI already active"}
        
        if voice_manager is None:
            raise HTTPException(status_code=500, detail="Voice manager not initialized")
        
        # Start voice AI in separate thread
        voice_thread = threading.Thread(target=voice_manager.start_voice_ai)
        voice_thread.daemon = True
        voice_thread.start()
        
        voice_active = True
        logger.info("Voice AI started successfully")
        
        return {"message": "Voice AI started successfully"}
        
    except Exception as e:
        logger.error(f"Failed to start voice AI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start voice AI: {str(e)}")

@app.post("/stop-voice")
async def stop_voice():
    """Stop the ElevenLabs voice AI"""
    global voice_active
    
    try:
        if not voice_active:
            return {"message": "Voice AI not active"}
        
        if voice_manager:
            voice_manager.stop_voice_ai()
        
        voice_active = False
        logger.info("Voice AI stopped successfully")
        
        return {"message": "Voice AI stopped successfully"}
        
    except Exception as e:
        logger.error(f"Failed to stop voice AI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop voice AI: {str(e)}")

@app.get("/voice-stream")
async def voice_stream():
    """Server-sent events stream for voice transcripts"""
    async def event_stream():
        while voice_active:
            try:
                # Check for transcripts
                if not transcript_queue.empty():
                    transcript_data = transcript_queue.get()
                    yield f"data: {json.dumps(transcript_data)}\n\n"
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in voice stream: {str(e)}")
                break
    
    return StreamingResponse(event_stream(), media_type="text/plain")

@app.get("/screenshot-triggers")
async def screenshot_triggers():
    """Endpoint to check for voice-triggered screenshot requests"""
    triggers = []
    while not screenshot_trigger_queue.empty():
        triggers.append(screenshot_trigger_queue.get())
    return {"triggers": triggers}

# Screenshot analysis endpoints (updated)
@app.post("/analyze-screenshot", response_model=AnalysisResult)
async def analyze_screenshot(request: AnalysisRequest):
    """Main endpoint for screenshot analysis"""
    try:
        logger.info(f"Received screenshot analysis request (voice_triggered: {request.voice_triggered})")
        
        if analyzer is None:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")
        
        temp_image_path = analyzer.save_temp_image(
            request.screenshot.image_data, 
            request.screenshot.filename
        )
        
        try:
            analysis_data = await analyzer.analyze_with_gemini(temp_image_path)
            
            save_success = await analyzer.save_to_supabase(
                analysis_data,
                request.screenshot.timestamp,
                request.user_id,
                request.voice_triggered
            )
            
            if not save_success:
                logger.warning("Failed to save to Supabase, but continuing with response")
            
            return AnalysisResult(
                analysis_data=analysis_data,
                timestamp=request.screenshot.timestamp,
                success=True
            )
            
        finally:
            analyzer.cleanup_temp_files()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze_screenshot: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return AnalysisResult(
            analysis_data={},
            timestamp=request.screenshot.timestamp,
            success=False,
            error_message=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health_status = {
            "status": "healthy", 
            "timestamp": datetime.now().isoformat(),
            "analyzer_initialized": analyzer is not None,
            "voice_manager_initialized": voice_manager is not None,
            "voice_active": voice_active,
            "gemini_configured": config.GEMINI_API_KEY is not None,
            "supabase_configured": config.SUPABASE_URL is not None and config.SUPABASE_KEY is not None,
            "elevenlabs_configured": config.ELEVENLABS_API_KEY is not None
        }
        
        # Test Supabase connection
        if analyzer:
            try:
                result = analyzer.supabase.table("screen_analyses").select("id").limit(1).execute()
                health_status["supabase_connection"] = "ok"
            except Exception as e:
                health_status["supabase_connection"] = f"error: {str(e)}"
        else:
            health_status["supabase_connection"] = "analyzer not initialized"
        
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/test-gemini")
async def test_gemini():
    """Test endpoint to verify Gemini API connectivity"""
    try:
        if analyzer is None:
            return {"error": "Analyzer not initialized"}
        
        test_model = genai.GenerativeModel('gemini-1.5-flash')
        response = test_model.generate_content("Say 'Hello, Gemini is working!'")
        
        return {
            "status": "success",
            "response": response.text,
            "model": "gemini-1.5-flash"
        }
    except Exception as e:
        logger.error(f"Gemini test failed: {str(e)}")
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.delete("/cleanup")
async def cleanup_temp_files():
    """Manually cleanup temporary files"""
    if analyzer:
        analyzer.cleanup_temp_files()
    return {"message": "Temporary files cleaned up"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Voice AI + Screenshot Analysis API",
        "version": "3.0.0",
        "description": "Integrated ElevenLabs voice AI with screenshot analysis system",
        "features": {
            "voice_ai": "ElevenLabs conversational AI",
            "screenshot_analysis": "Gemini vision API analysis",
            "voice_triggered_screenshots": "Say 'take a screenshot' for immediate capture",
            "automatic_cleanup": "Screenshots deleted after processing"
        },
        "endpoints": {
            "POST /start-voice": "Start ElevenLabs voice AI",
            "POST /stop-voice": "Stop voice AI",
            "GET /voice-stream": "Server-sent events for voice transcripts",
            "GET /screenshot-triggers": "Check for voice-triggered screenshot requests",
            "POST /analyze-screenshot": "Analyze screenshot with Gemini",
            "GET /health": "Health check"
        }
    }

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    global voice_active
    voice_active = False
    if voice_manager:
        voice_manager.stop_voice_ai()
    if analyzer:
        analyzer.cleanup_temp_files()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)