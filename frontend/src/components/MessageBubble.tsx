import type { Message } from "../types";

const RESOLUTION_LABELS: Record<string, string> = {
  classifier: "classifier",
  follow_up: "follow-up (context)",
  clarification: "answered clarification",
  fallback: "fallback",
  restricted: "role-restricted",
};

export default function MessageBubble({ message, showDetails }: { message: Message; showDetails: boolean }) {
  const { from, text, meta, error } = message;
  return (
    <div className={`row ${from}`}>
      {from === "bot" && (
        <div className="avatar" aria-hidden="true">
          EU
        </div>
      )}
      <div className={`bubble ${from}${error ? " error" : ""}`}>
        <p>{text}</p>
        {showDetails && meta && (
          <div className="meta">
            <span>{meta.intent}</span>
            <span>{Math.round(meta.confidence * 100)}%</span>
            <span>{RESOLUTION_LABELS[meta.resolved_by] ?? meta.resolved_by}</span>
          </div>
        )}
      </div>
    </div>
  );
}
