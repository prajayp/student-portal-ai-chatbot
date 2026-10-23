import { useEffect, useRef } from "react";
import type { Message } from "../types";
import MessageBubble from "./MessageBubble";
import TypingIndicator from "./TypingIndicator";

interface Props {
  messages: Message[];
  typing: boolean;
  showDetails: boolean;
}

export default function MessageList({ messages, typing, showDetails }: Props) {
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, typing]);

  return (
    <div className="messages" role="log" aria-live="polite">
      {messages.map((m) => (
        <MessageBubble key={m.id} message={m} showDetails={showDetails} />
      ))}
      {typing && <TypingIndicator />}
      <div ref={end} />
    </div>
  );
}
