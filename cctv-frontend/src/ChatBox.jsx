import React, { useState } from "react";

const API_BASE_URL = "http://127.0.0.1:5001";

export default function ChatBox() {
  const [messages, setMessages] = useState([
    { role: "system", text: "Hello 👋 Ask me anything!" },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const newMessage = { role: "user", text: input };
    setMessages((prev) => [...prev, newMessage]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: input }),
      });
      const data = await res.json();

      setMessages((prev) => [
        ...prev,
        { role: "system", text: data.answer || "No response." },
      ]);
    } catch (err) {
      console.error("Chat request failed:", err);
      setMessages((prev) => [
        ...prev,
        { role: "system", text: "⚠️ Error contacting server." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-900 p-4">
      <div className="w-full max-w-2xl bg-gray-800 rounded-2xl shadow-2xl flex flex-col overflow-hidden border border-gray-700">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex ${
                msg.role === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`px-4 py-2 rounded-xl max-w-xs sm:max-w-md text-sm ${
                  msg.role === "user"
                    ? "bg-teal-600 text-white rounded-br-none"
                    : "bg-gray-700 text-gray-100 rounded-bl-none"
                }`}
              >
                {msg.text}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="px-4 py-2 rounded-xl bg-gray-700 text-gray-300 animate-pulse">
                Typing...
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <form
          onSubmit={sendMessage}
          className="flex items-center border-t border-gray-700 bg-gray-800 p-4"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question..."
            className="flex-1 bg-gray-700 text-white rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-teal-500"
          />
          <button
            type="submit"
            className="ml-3 bg-teal-600 hover:bg-teal-700 text-white font-bold px-4 py-2 rounded-lg transition"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
