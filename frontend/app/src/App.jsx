import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Camera, Play, Pause, AlertCircle, Database, Eye, Settings, Mic, MicOff, MessageCircle } from 'lucide-react';

const ScreenshotAnalysisSystem = () => {
  const [isCapturing, setIsCapturing] = useState(false);
  const [isVoiceActive, setIsVoiceActive] = useState(false);
  const [analysisResults, setAnalysisResults] = useState([]);
  const [voiceTranscripts, setVoiceTranscripts] = useState([]);
  const [status, setStatus] = useState('');
  const [captureInterval, setCaptureInterval] = useState(30); // seconds
  const [backendUrl, setBackendUrl] = useState('http://localhost:8000');
  const intervalRef = useRef(null);
  const streamRef = useRef(null);

  // Load analysis results from memory on component mount
  useEffect(() => {
    const savedResults = JSON.parse(sessionStorage.getItem('analysisResults') || '[]');
    const savedTranscripts = JSON.parse(sessionStorage.getItem('voiceTranscripts') || '[]');
    setAnalysisResults(savedResults);
    setVoiceTranscripts(savedTranscripts);
  }, []);

  // Save analysis results to memory
  const saveResultsToMemory = (newResults) => {
    sessionStorage.setItem('analysisResults', JSON.stringify(newResults));
  };

  // Save voice transcripts to memory
  const saveTranscriptsToMemory = (newTranscripts) => {
    sessionStorage.setItem('voiceTranscripts', JSON.stringify(newTranscripts));
  };

  // Request screen capture permission and get stream
  const getScreenStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: {
          mediaSource: 'screen',
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        }
      });
      return stream;
    } catch (error) {
      setStatus(`Screen capture permission denied: ${error.message}`);
      return null;
    }
  };

  // Capture screenshot from stream
  const captureScreenshot = useCallback(async () => {
    if (!streamRef.current) {
      setStatus('No screen stream available');
      return null;
    }

    try {
      // Create video element to capture frame
      const video = document.createElement('video');
      video.srcObject = streamRef.current;
      video.play();

      // Wait for video to load
      await new Promise((resolve) => {
        video.onloadedmetadata = resolve;
      });

      // Create canvas and capture frame
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0);

      // Convert to base64
      const base64Data = canvas.toDataURL('image/png');
      
      // Clean up
      video.srcObject = null;

      return {
        filename: `screenshot_${Date.now()}.png`,
        image_data: base64Data,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      setStatus(`Screenshot capture failed: ${error.message}`);
      return null;
    }
  }, []);

  // Send screenshot to backend for analysis
  const sendToBackend = async (screenshotData, isVoiceTriggered = false) => {
    try {
      setStatus(`${isVoiceTriggered ? 'Voice-triggered s' : 'S'}ending screenshot for analysis...`);
      
      const response = await fetch(`${backendUrl}/analyze-screenshot`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          screenshot: screenshotData,
          voice_triggered: isVoiceTriggered
        })
      });

      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.success) {
        // Add analysis result to state
        const newResult = {
          id: Date.now(),
          analysis: result.analysis_data,
          timestamp: result.timestamp,
          voice_triggered: isVoiceTriggered
        };
        
        const updatedResults = [...analysisResults, newResult];
        setAnalysisResults(updatedResults);
        saveResultsToMemory(updatedResults);
        
        setStatus(`Analysis completed at ${new Date(result.timestamp).toLocaleTimeString()}${isVoiceTriggered ? ' (Voice triggered)' : ''}`);
      } else {
        setStatus(`Analysis failed: ${result.error_message}`);
      }
    } catch (error) {
      setStatus(`Backend communication failed: ${error.message}`);
    }
  };

  // Process screenshot (capture and analyze)
  const processScreenshot = useCallback(async (isVoiceTriggered = false) => {
    const screenshot = await captureScreenshot();
    if (screenshot) {
      await sendToBackend(screenshot, isVoiceTriggered);
      // Screenshot is automatically deleted by backend after processing
    }
  }, [captureScreenshot, sendToBackend, analysisResults]);

  // Start voice AI and screenshot system
  const startSystem = useCallback(async () => {
    try {
      setStatus('Starting voice AI and screenshot system...');
      
      // Start voice AI
      const voiceResponse = await fetch(`${backendUrl}/start-voice`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (!voiceResponse.ok) {
        throw new Error(`Voice AI failed to start: ${voiceResponse.status}`);
      }

      setIsVoiceActive(true);
      
      // Start screenshot capture
      const stream = await getScreenStream();
      if (!stream) return;

      streamRef.current = stream;
      setIsCapturing(true);
      setStatus(`System started - Voice AI active, capturing screenshots every ${captureInterval} seconds`);

      // Initial capture
      await processScreenshot();

      // Set up interval for periodic capture
      intervalRef.current = setInterval(async () => {
        await processScreenshot();
      }, captureInterval * 1000);

      // Handle stream end (user stops sharing)
      stream.getVideoTracks()[0].onended = () => {
        stopSystem();
        setStatus('Screen sharing stopped by user');
      };

      // Start listening for voice transcripts
      startVoiceListener();

    } catch (error) {
      setStatus(`Failed to start system: ${error.message}`);
    }
  }, [captureInterval, processScreenshot]);

  // Start voice transcript listener
  const startVoiceListener = useCallback(() => {
    const eventSource = new EventSource(`${backendUrl}/voice-stream`);
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'transcript') {
        const newTranscript = {
          id: Date.now(),
          text: data.text,
          timestamp: new Date().toISOString(),
          isUser: data.isUser
        };
        
        const updatedTranscripts = [...voiceTranscripts, newTranscript];
        setVoiceTranscripts(updatedTranscripts);
        saveTranscriptsToMemory(updatedTranscripts);
        
        // Check if user said "take a screenshot"
        if (data.isUser && data.text.toLowerCase().includes('take a screenshot')) {
          processScreenshot(true); // Voice-triggered screenshot
        }
      }
    };

    eventSource.onerror = (error) => {
      console.error('Voice stream error:', error);
      eventSource.close();
    };

    return eventSource;
  }, [voiceTranscripts, processScreenshot]);

  // Stop voice AI and screenshot system
  const stopSystem = useCallback(async () => {
    try {
      // Stop voice AI
      await fetch(`${backendUrl}/stop-voice`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      setIsVoiceActive(false);

      // Stop screenshot capture
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }

      setIsCapturing(false);
      setStatus('System stopped - Voice AI and screenshot capture disabled');
    } catch (error) {
      setStatus(`Failed to stop system: ${error.message}`);
    }
  }, []);

  // Clear all results
  const clearResults = useCallback(() => {
    setAnalysisResults([]);
    setVoiceTranscripts([]);
    saveResultsToMemory([]);
    saveTranscriptsToMemory([]);
    setStatus('All results cleared');
  }, []);

  // Manual single capture
  const manualCapture = useCallback(async () => {
    if (!isCapturing) {
      setStatus('System not running. Start the system first.');
      return;
    }

    await processScreenshot(false);
  }, [isCapturing, processScreenshot]);

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gray-50 min-h-screen">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-6 flex items-center">
          <Camera className="mr-3 text-blue-600" />
          Voice AI + Screenshot Analysis System
        </h1>

        {/* Status Bar */}
        {status && (
          <div className="mb-6 p-3 bg-blue-50 border-l-4 border-blue-400 rounded">
            <p className="text-blue-800 flex items-center">
              <AlertCircle className="mr-2 h-4 w-4" />
              {status}
            </p>
          </div>
        )}

        {/* Configuration Section */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              <Settings className="inline mr-1 h-4 w-4" />
              Capture Interval (seconds)
            </label>
            <input
              type="number"
              value={captureInterval}
              onChange={(e) => setCaptureInterval(Math.max(5, parseInt(e.target.value) || 30))}
              min="5"
              max="300"
              disabled={isCapturing}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Backend URL
            </label>
            <input
              type="url"
              value={backendUrl}
              onChange={(e) => setBackendUrl(e.target.value)}
              placeholder="http://localhost:8000"
              disabled={isCapturing}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
            />
          </div>

          <div className="flex items-end">
            <div className="text-sm text-gray-600">
              <p className="font-medium">System Status:</p>
              <div className="flex items-center gap-4 mt-1">
                <span className={`flex items-center ${isVoiceActive ? 'text-green-600' : 'text-gray-400'}`}>
                  {isVoiceActive ? <Mic className="h-4 w-4 mr-1" /> : <MicOff className="h-4 w-4 mr-1" />}
                  Voice AI
                </span>
                <span className={`flex items-center ${isCapturing ? 'text-green-600' : 'text-gray-400'}`}>
                  <Camera className="h-4 w-4 mr-1" />
                  Capture
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Control Section */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">System Controls</h2>
          <div className="flex gap-4 flex-wrap">
            {!isCapturing ? (
              <button
                onClick={startSystem}
                className="px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center font-medium"
              >
                <Play className="mr-2 h-4 w-4" />
                Start Voice AI + Screenshot System
              </button>
            ) : (
              <>
                <button
                  onClick={stopSystem}
                  className="px-6 py-3 bg-red-600 text-white rounded-md hover:bg-red-700 flex items-center font-medium"
                >
                  <Pause className="mr-2 h-4 w-4" />
                  Stop System
                </button>
                <button
                  onClick={manualCapture}
                  className="px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center font-medium"
                >
                  <Camera className="mr-2 h-4 w-4" />
                  Manual Screenshot
                </button>
              </>
            )}
            
            {(analysisResults.length > 0 || voiceTranscripts.length > 0) && (
              <button
                onClick={clearResults}
                className="px-4 py-3 bg-gray-600 text-white rounded-md hover:bg-gray-700 flex items-center"
              >
                Clear All Results
              </button>
            )}
          </div>

          {isCapturing && (
            <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-md">
              <p className="text-green-800 flex items-center">
                <div className="animate-pulse w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                System active - Voice AI listening, screenshots every {captureInterval} seconds
              </p>
              <p className="text-sm text-green-600 mt-1">
                💡 Say "take a screenshot" to capture immediately
              </p>
            </div>
          )}
        </div>

        {/* Voice Transcripts */}
        {voiceTranscripts.length > 0 && (
          <div className="mb-8">
            <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
              <MessageCircle className="mr-2" />
              Voice Conversation ({voiceTranscripts.length})
            </h2>
            <div className="space-y-2 max-h-64 overflow-y-auto bg-gray-50 p-4 rounded-lg">
              {voiceTranscripts.slice().reverse().map((transcript) => (
                <div key={transcript.id} className={`flex ${transcript.isUser ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                    transcript.isUser 
                      ? 'bg-blue-500 text-white' 
                      : 'bg-white border border-gray-200'
                  }`}>
                    <p className="text-sm">{transcript.text}</p>
                    <p className={`text-xs mt-1 ${transcript.isUser ? 'text-blue-100' : 'text-gray-500'}`}>
                      {new Date(transcript.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Analysis Results */}
        {analysisResults.length > 0 && (
          <div>
            <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
              <Database className="mr-2" />
              Screenshot Analysis Results ({analysisResults.length})
            </h2>
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {analysisResults.slice().reverse().map((result) => (
                <div key={result.id} className="bg-gray-50 rounded-lg p-4 border">
                  <div className="flex justify-between items-start mb-3">
                    <h3 className="font-medium text-gray-800 flex items-center">
                      Analysis #{analysisResults.length - analysisResults.indexOf(result)}
                      {result.voice_triggered && (
                        <span className="ml-2 px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full flex items-center">
                          <Mic className="h-3 w-3 mr-1" />
                          Voice
                        </span>
                      )}
                    </h3>
                    <span className="text-xs text-gray-500">
                      {new Date(result.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <div className="bg-white rounded p-3">
                    <div className="grid md:grid-cols-2 gap-4 mb-3">
                      <div>
                        <p className="text-sm font-medium text-gray-700">Application:</p>
                        <p className="text-sm text-gray-900">{result.analysis.application_identification?.application_name || 'Unknown'}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-700">Type:</p>
                        <p className="text-sm text-gray-900">{result.analysis.application_identification?.application_type || 'Unknown'}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-700">Confidence:</p>
                        <p className="text-sm text-gray-900">{result.analysis.application_identification?.confidence_score || 'N/A'}</p>
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-700">Current Activity:</p>
                        <p className="text-sm text-gray-900">{result.analysis.user_context?.current_activity || 'Unknown'}</p>
                      </div>
                    </div>
                    <details className="text-xs">
                      <summary className="cursor-pointer text-blue-600 hover:text-blue-800 flex items-center">
                        <Eye className="mr-1 h-3 w-3" />
                        View Full Analysis JSON
                      </summary>
                      <pre className="mt-2 p-2 bg-gray-100 rounded overflow-auto max-h-64 text-xs">
                        {JSON.stringify(result.analysis, null, 2)}
                      </pre>
                    </details>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {analysisResults.length === 0 && voiceTranscripts.length === 0 && !isCapturing && (
          <div className="text-center py-12">
            <div className="flex justify-center items-center mb-4">
              <Camera className="h-12 w-12 text-gray-400 mr-4" />
              <Mic className="h-12 w-12 text-gray-400" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">Ready for Voice AI + Screenshot Analysis</h3>
            <p className="text-gray-600 mb-4">Click "Start Voice AI + Screenshot System" to begin</p>
            <div className="text-sm text-gray-500 space-y-1">
              <p>• Voice AI will listen for commands and conversations</p>
              <p>• Screenshots will be captured automatically every {captureInterval} seconds</p>
              <p>• Say "take a screenshot" for immediate capture</p>
              <p>• Screenshots are analyzed and then automatically deleted</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ScreenshotAnalysisSystem;