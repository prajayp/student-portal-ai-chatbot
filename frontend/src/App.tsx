import { useCallback, useEffect, useRef, useState } from "react";
import { newId, sendMessage } from "./api";
import Composer from "./components/Composer";
import ContextBar from "./components/ContextBar";
import MessageList from "./components/MessageList";
import RoleSelector from "./components/RoleSelector";
import SuggestionChips from "./components/SuggestionChips";
import type { Message, Role } from "./types";

export default function App() {
  const [role, setRole] = useState<Role>("student");
  const [sessionId, setSessionId] = useState(newId);
  const [messages, setMessages] = useState<Message[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [slots, setSlots] = useState<Record<string, string | number>>({});
  const [typing, setTyping] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const greeted = useRef<string | null>(null);

  const ask = useCallback(
    async (text: string, { silent = false } = {}) => {
      if (!silent) {
        setMessages((m) => [...m, { id: newId(), from: "user", text }]);
      }
      setSuggestions([]);
      setTyping(true);
      try {
        const { reply, suggestions: next, ...meta } = await sendMessage(sessionId, role, text);
        setMessages((m) => [...m, { id: newId(), from: "bot", text: reply, meta }]);
        setSuggestions(next);
        setSlots(meta.slots);
      } catch {
        setMessages((m) => [
          ...m,
          {
            id: newId(),
            from: "bot",
            error: true,
            text: "I couldn't reach the server. Is the backend running on port 8000?",
          },
        ]);
      } finally {
        setTyping(false);
      }
    },
    [role, sessionId],
  );

  // Greet once per session; the guard also keeps React StrictMode from greeting twice.
  useEffect(() => {
    if (greeted.current === sessionId) return;
    greeted.current = sessionId;
    void ask("hi", { silent: true });
  }, [ask, sessionId]);

  const changeRole = (next: Role) => {
    if (next === role) return;
    setRole(next);
    setMessages([]);
    setSlots({});
    setSuggestions([]);
    setSessionId(newId());
  };

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="logo" aria-hidden="true">EU</div>
          <div>
            <h1>Portal Assistant</h1>
            <p>Example University</p>
          </div>
        </div>
        <RoleSelector value={role} onChange={changeRole} disabled={typing} />
      </header>

      <main className="chat">
        <MessageList messages={messages} typing={typing} showDetails={showDetails} />
      </main>

      <footer className="footer">
        <ContextBar slots={slots} />
        <SuggestionChips suggestions={suggestions} onPick={ask} disabled={typing} />
        <Composer onSend={ask} disabled={typing} />
        <label className="details-toggle">
          <input type="checkbox" checked={showDetails} onChange={(e) => setShowDetails(e.target.checked)} />
          Show NLP details
        </label>
      </footer>
    </div>
  );
}
