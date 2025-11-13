import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";
import LogPanel from "./LogPanel.jsx";
import ChatBox from "./ChatBox.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
    <LogPanel />
    <ChatBox />
  </StrictMode>
);
