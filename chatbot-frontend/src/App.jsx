import { useEffect, useMemo, useRef, useState } from "react";
import { sendChat } from "./api/chat";
import "./App.css";

import bg from "./assets/masjid-bg.jpg"; // <- your masjid image

const SYSTEM_PROMPT =
  "You are a helpful Tafsir assistant. Answer clearly and concisely.";

export default function App() {
  const [messages, setMessages] = useState([{ role: "system", content: SYSTEM_PROMPT }]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  const visible = useMemo(() => messages.filter((m) => m.role !== "system"), [messages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [visible.length, loading]);

  async function handleSend(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    const userMsg = { role: "user", content: text };
    const next = [...messages, userMsg];

    setMessages(next); // user bubble appears immediately
    setInput("");
    setLoading(true);

    try {
      const data = await sendChat(next);
      setMessages((prev) => [...prev, { role: "assistant", content: data.reply }]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `⚠️ ${err.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function newChat() {
    setMessages([{ role: "system", content: SYSTEM_PROMPT }]);
  }

  return (
    <div className="page" style={{ "--bg": `url(${bg})` }}>
      <div className="layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sb-brand">
            <div className="sb-mark">ٱلنُّور</div>
            <div className="sb-title">Noor</div>
            <div className="sb-sub">AI Tafsir Assistant</div>
          </div>

          <button className="sb-new" onClick={newChat} disabled={loading}>
            + New Chat
          </button>

          <div className="sb-bottom">
            <div className="sb-toggle">
              <span>Scholar Mode</span>
              <span className="toggle-pill on">
                <span className="dot" />
              </span>
            </div>

            <div className="sb-links">
              <button className="sb-link">Settings</button>
              <button className="sb-link">About</button>
            </div>
          </div>
        </aside>

        {/* Main "big bubble" */}
        <main className="main">
          <section className="mainBubble">
            {/* Top header area inside the bubble */}
            <header className="bubbleHeader">
              <div className="bubbleHeaderLeft">
                <div className="titleRow">
                  <span className="titleAr">ٱلنُّور</span>
                  <span className="titleEn">Noor</span>
                </div>
                <div className="titleSub">Ask a Tafsir Question</div>

                <div className="chips">
                  <span className="chip">🕋 Qur’an &amp; Tafseer Focused</span>
                  <span className="chip">✅ Verified Sources</span>
                </div>
              </div>

              
            </header>

            {/* Chat scroll area */}
            <div className="chat">
              {visible.map((m, i) => (
                <div
                  key={i}
                  className={`bubbleMsg ${m.role === "user" ? "bubbleUser" : "bubbleBot"}`}
                >
                  {m.content}
                </div>
              ))}

              {loading && <div className="bubbleMsg bubbleBot bubbleTyping">Typing…</div>}
              <div ref={bottomRef} />
            </div>

            {/* Input bar inside the bubble */}
            <form className="composer" onSubmit={handleSend}>
              <input
                className="composerInput"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask a question about any part of the Qur’an..."
                disabled={loading}
              />
              <button className="sendBtn" type="submit" disabled={loading || !input.trim()}>
                ➤
              </button>
            </form>
          </section>
        </main>
      </div>
    </div>
  );
}
