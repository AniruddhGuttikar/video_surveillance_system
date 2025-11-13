import React, { useState, useEffect, useRef } from "react";

// --- Configuration ---
const API_BASE_URL = "http://127.0.0.1:5001"; // Backend API URL

// --- SVG Icons (Unchanged) ---
const UploadIcon = () => (
  <svg
    xmlns="http://www.w.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="mr-2 h-5 w-5"
  >
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" />
    <line x1="12" x2="12" y1="3" y2="15" />
  </svg>
);
const FileVideoIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="mr-2 h-5 w-5"
  >
    <path d="M14.5 4H5a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V7.5L14.5 4z" />
    <polyline points="14 2 14 8 20 8" />
    <path d="m10 15.5 4 2.5v-6l-4 2.5" />
    <path d="M8 12v8" />
  </svg>
);
const AlertTriangleIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="h-8 w-8 text-red-400"
  >
    <path d="m21.73 18-8-14a2 2 0 0 0-3.46 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
    <line x1="12" x2="12" y1="9" y2="13" />
    <line x1="12" x2="12.01" y1="17" y2="17" />
  </svg>
);
const CheckCircleIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="h-8 w-8 text-green-400"
  >
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
    <polyline points="22 4 12 14.01 9 11.01" />
  </svg>
);

// --- Main App Component ---
export default function App() {
  const [view, setView] = useState("main"); // 'main', 'realtime'
  const [status, setStatus] = useState("System Standby");
  const [results, setResults] = useState([]);
  const [phoneNumber, setPhoneNumber] = useState("+91");
  const [simulationFile, setSimulationFile] = useState(null);

  // Fetch system status periodically
  useEffect(() => {
    const interval = setInterval(() => {
      fetch(`${API_BASE_URL}/status`)
        .then((res) => res.json())
        .then((data) => setStatus(data.status))
        .catch((err) => console.error("Failed to fetch status:", err));
    }, 20000);
    return () => clearInterval(interval);
  }, []);

  // Fetch results when the main view is shown
  useEffect(() => {
    if (view === "main") {
      fetch(`${API_BASE_URL}/results`)
        .then((res) => res.json())
        .then((data) => setResults(data.files || []))
        .catch((err) => console.error("Failed to fetch results:", err));
    }
  }, [view]);

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("video", file);

    try {
      await fetch(`${API_BASE_URL}/upload`, { method: "POST", body: formData });
      setStatus(`Processing: ${file.name}`);
    } catch (error) {
      console.error("Upload failed:", error);
      setStatus("Error: Upload failed.");
    }
  };

  const handleStartSimulation = async (event) => {
    event.preventDefault(); // Prevent default form submission
    if (!simulationFile) {
      alert("Please select a video file to simulate.");
      return;
    }
    if (!phoneNumber || !phoneNumber.match(/^\+[0-9]{10,15}$/)) {
      alert(
        "Please enter a valid phone number with country code (e.g., +1234567890)."
      );
      return;
    }

    const formData = new FormData();
    formData.append("demo_video", simulationFile);
    formData.append("phone_number", phoneNumber);

    try {
      await fetch(`${API_BASE_URL}/start_simulation`, {
        method: "POST",
        body: formData,
      });
      setView("realtime");
    } catch (error) {
      console.error("Failed to start simulation:", error);
      setStatus("Error: Could not start simulation.");
    }
  };

  const stopRealtime = async () => {
    try {
      await fetch(`${API_BASE_URL}/stop_realtime`, { method: "POST" });
      setView("main");
      setStatus("System Standby");
    } catch (error) {
      console.error("Failed to stop real-time feed:", error);
    }
  };

  if (view === "realtime") {
    return (
      <RealtimeView
        status={status}
        phoneNumber={phoneNumber}
        onStop={stopRealtime}
      />
    );
  }

  return (
    <MainView
      status={status}
      results={results}
      onFileUpload={handleFileUpload}
      onStartSimulation={handleStartSimulation}
      phoneNumber={phoneNumber}
      setPhoneNumber={setPhoneNumber}
      simulationFile={simulationFile} // Pass the simulationFile state
      setSimulationFile={setSimulationFile}
    />
  );
}

// --- Sub-components for different views ---
const MainView = ({
  status,
  results,
  onFileUpload,
  onStartSimulation,
  phoneNumber,
  setPhoneNumber,
  simulationFile,
  setSimulationFile,
}) => {
  const fileInputRef = useRef(null);
  const simFileInputRef = useRef(null);

  const handleSimFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSimulationFile(file);
    }
  };

  return (
    <div className="bg-gray-900 text-gray-200 sm:p-8">
      {/* <div className="max-w-4xl mx-auto"> */}
      <Header />
      <StatusDisplay status={status} />
      <div className="grid md:grid-cols-2 gap-8 mt-8">
        <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
          <h2 className="text-xl font-bold text-cyan-400 mb-4">
            Offline Video Analysis
          </h2>
          <p className="text-gray-400 mb-4">
            Upload a video file. The system will process it and save anomaly
            clips.
          </p>
          <input
            type="file"
            ref={fileInputRef}
            onChange={onFileUpload}
            className="hidden"
            accept="video/*"
          />
          <button
            onClick={() => fileInputRef.current.click()}
            className="w-full flex items-center justify-center bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300"
          >
            <UploadIcon /> Upload & Process Video
          </button>
        </div>

        <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
          <h2 className="text-xl font-bold text-teal-400 mb-4">
            Live Monitoring Simulation
          </h2>
          <p className="text-gray-400 mb-4">
            Upload a video to simulate a live feed and receive SMS alerts.
          </p>
          <form onSubmit={onStartSimulation}>
            <input
              type="tel"
              value={phoneNumber}
              onChange={(e) => setPhoneNumber(e.target.value)}
              placeholder="+1234567890"
              className="w-full bg-gray-700 border border-gray-600 rounded-lg p-3 mb-4 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-teal-500"
              required
            />
            <input
              type="file"
              ref={simFileInputRef}
              onChange={handleSimFileChange}
              className="hidden"
              accept="video/*"
            />
            <button
              type="button"
              onClick={() => simFileInputRef.current.click()}
              className="w-full flex items-center justify-center bg-gray-600 hover:bg-gray-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300 mb-4"
            >
              <FileVideoIcon />{" "}
              {simulationFile
                ? `Selected: ${simulationFile.name}`
                : "Select Simulation Video"}
            </button>
            <button
              type="submit"
              className="w-full flex items-center justify-center bg-teal-600 hover:bg-teal-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300"
            >
              Start Simulation
            </button>
          </form>
        </div>
      </div>
      {/* <ResultsPanel results={results} /> */}
      {/* </div> */}
    </div>
  );
};

const RealtimeView = ({ status, phoneNumber, onStop }) => {
  const isAnomaly = status.includes("ANOMALY");
  return (
    <div className="bg-gray-900 text-gray-200 min-h-screen p-4 sm:p-8 flex flex-col items-center">
      <div className="w-full max-w-4xl">
        <h1 className="text-3xl font-bold text-center text-red-500 mb-2">
          Live Simulation Active
        </h1>
        <p className="text-center text-gray-400 mb-4">
          Alerts will be sent to: <strong>{phoneNumber}</strong>
        </p>
        <div className="bg-black rounded-lg shadow-2xl border-2 border-gray-700 overflow-hidden">
          <img
            src={`${API_BASE_URL}/video_feed?t=${new Date().getTime()}`}
            alt="Live Feed"
            className="w-full"
          />
        </div>
        <div
          className={`mt-6 p-4 rounded-lg text-center text-2xl font-mono transition-all duration-300 ${
            isAnomaly
              ? "bg-red-900 border-red-700 text-red-200"
              : "bg-green-900 border-green-700 text-green-200"
          } border-2`}
        >
          {status}
        </div>
        <div className="text-center mt-6">
          <button
            onClick={onStop}
            className="bg-gray-600 hover:bg-gray-700 text-white font-bold py-3 px-8 rounded-lg transition duration-300"
          >
            Stop Simulation
          </button>
        </div>
      </div>
    </div>
  );
};

// --- Helper Components (Unchanged) ---
const Header = () => (
  <header className="text-center mb-8">
    <h1 className="text-4xl sm:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-teal-500">
      AI Anomaly Detection System
    </h1>
    <p className="text-gray-400 mt-2">Powered by VideoMAE & YOLOv8</p>
  </header>
);

const StatusDisplay = ({ status }) => {
  const lowerCaseStatus = status.toLowerCase();
  const isAnomaly =
    lowerCaseStatus.includes("anomaly") || lowerCaseStatus.includes("error");
  const isProcessing =
    (lowerCaseStatus.includes("processing") ||
      lowerCaseStatus.includes("step")) &&
    !lowerCaseStatus.includes("complete");

  return (
    <div
      className={`p-4 rounded-lg text-center text-lg font-semibold border ${
        isAnomaly
          ? "bg-red-900/50 border-red-700 text-red-300"
          : isProcessing
          ? "bg-blue-900/50 border-blue-700 text-blue-300"
          : "bg-gray-800 border-gray-700 text-gray-300"
      } transition-all duration-300 flex items-center justify-center`}
    >
      {isAnomaly ? <AlertTriangleIcon /> : <CheckCircleIcon />}
      <span className="ml-3">{status}</span>
      {isProcessing && (
        <div className="ml-4 w-5 h-5 border-4 border-dashed rounded-full animate-spin border-blue-400"></div>
      )}
    </div>
  );
};

const ResultsPanel = ({ results }) => (
  <div className="mt-8 bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
    <h2 className="text-xl font-bold text-cyan-400 mb-4">
      Processed Clips & Logs
    </h2>
    <div className="max-h-60 overflow-y-auto pr-2">
      {results.length > 0 ? (
        <ul className="space-y-2">
          {results.map((file) => (
            <li
              key={file}
              className="bg-gray-700 p-3 rounded-md flex justify-between items-center"
            >
              <span className="text-gray-300">{file}</span>
              <a
                href={`${API_BASE_URL}/results/${file}`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-cyan-400 hover:text-cyan-300 font-semibold"
              >
                {file.endsWith(".txt") ? "View Log" : "Play Clip"}
              </a>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-gray-500 text-center">
          No processed files yet. Upload a video to begin.
        </p>
      )}
    </div>
  </div>
);
