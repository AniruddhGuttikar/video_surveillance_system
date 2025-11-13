import React, { useState, useEffect } from "react";

const API_BASE_URL = "http://127.0.0.1:5001"; // Backend API URL

const LogPanel = () => {
  const severityColors = {
    INFO: "bg-green-100 text-green-800",
    WARN: "bg-yellow-100 text-yellow-800",
    ALERT: "bg-red-100 text-red-800",
  };
  
  const [logs, setLogs] = useState([]);
  const [lastCount, setLastCount] = useState(0);

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        // Check if there are new clips
        const countRes = await fetch(`${API_BASE_URL}/logs/count`);
        const countData = await countRes.json();
        
        if (countData.count > lastCount) {
          // Fetch ALL logs (not just new ones)
          const logsRes = await fetch(`${API_BASE_URL}/logs/all`);
          const logsData = await logsRes.json();
          
          if (logsData.logs && Array.isArray(logsData.logs)) {
            setLogs(logsData.logs);
            setLastCount(countData.count);
          }
        }
      } catch (err) {
        console.error("Log polling failed:", err);
      }
    };

    // Fetch immediately on mount
    fetchLogs();
    
    // Then poll every 5 seconds
    const interval = setInterval(fetchLogs, 5000);
    return () => clearInterval(interval);
  }, [lastCount]);

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700 max-h-96 overflow-y-auto">
      <h2 className="text-xl font-bold text-pink-400 mb-4">Live Event Logs</h2>
      {logs.length > 0 ? (
        <ul className="space-y-3">
          {logs.map((log, idx) => (
            <li
              key={idx}
              className={`p-4 rounded-lg shadow-md ${
                severityColors[log.severity] || "bg-gray-200 text-gray-800"
              }`}
            >
              <div className="flex justify-between items-start">
                <div>
                  <span className="font-mono text-xs opacity-70 block">
                    Clip #{log.clip_number} • {log.timestamp}
                  </span>
                  <span className="font-bold text-sm">{log.severity}</span>
                </div>
              </div>
              <p className="mt-2 text-sm">{log.summary}</p>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-gray-500 text-center">
          No events detected yet. Upload or simulate a video to begin.
        </p>
      )}
    </div>
  );
};
export default LogPanel;