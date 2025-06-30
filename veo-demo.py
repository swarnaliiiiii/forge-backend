"""
Standalone ElevenLabs Voice AI Demo
Test file to verify ElevenLabs functionality before integration
"""

import os
import signal
import time
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation, ConversationInitiationData
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

# Load environment variables from .env file
load_dotenv()

agent_id = os.getenv("AGENT_ID")
if not agent_id:
    raise ValueError("AGENT_ID environment variable is not set.")

api_key = os.getenv("ELEVENLABS_API_KEY")
if not api_key:
    raise ValueError("ELEVENLABS_API_KEY environment variable is not set.")

elevenlabs = ElevenLabs(api_key=api_key)

# Test callbacks to demonstrate integration points
def user_transcript_callback(transcript):
    """Callback for user speech - this is where screenshot triggers would be detected"""
    print(f"User: {transcript}")
    
    # Simulate screenshot trigger detection
    if "take a screenshot" in transcript.lower():
        print("🔥 SCREENSHOT TRIGGER DETECTED! This would trigger a screenshot in the main system.")

def agent_response_callback(response):
    """Callback for agent response"""
    print(f"Agent: {response}")

def agent_response_correction_callback(original, corrected):
    """Callback for agent response corrections"""
    print(f"Agent correction: {original} -> {corrected}")

# Dynamic variables for the conversation
dynamic_vars = {
    "user_name": "Test User",
    "user_id": "test_user_1",
}

# Conversation configuration
config = ConversationInitiationData(
    dynamic_variables=dynamic_vars
)

# Create conversation with callbacks
conversation = Conversation(
    elevenlabs,
    agent_id,
    config=config,
    # Assume auth is required when API_KEY is set
    requires_auth=bool(api_key),
    # Use the default audio interface
    audio_interface=DefaultAudioInterface(),
    # Callbacks that demonstrate integration points
    callback_agent_response=agent_response_callback,
    callback_agent_response_correction=agent_response_correction_callback,
    callback_user_transcript=user_transcript_callback,
    # Uncomment to see latency measurements
    # callback_latency_measurement=lambda latency: print(f"Latency: {latency}ms"),
)

print("🎤 Starting ElevenLabs Voice AI Demo...")
print("💡 Try saying 'take a screenshot' to see the trigger detection!")
print("🛑 Press Ctrl+C to stop")
print("-" * 50)

try:
    # Start the conversation session
    conversation.start_session()
    
    # Keep the conversation alive
    while True:
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\n🛑 Stopping conversation...")
    conversation.end_session()
    print("✅ Demo ended")
except Exception as e:
    print(f"❌ Error: {e}")
    conversation.end_session()

# Handle graceful shutdown
def signal_handler(sig, frame):
    print("\n🛑 Received shutdown signal...")
    conversation.end_session()
    exit(0)

signal.signal(signal.SIGINT, signal_handler)