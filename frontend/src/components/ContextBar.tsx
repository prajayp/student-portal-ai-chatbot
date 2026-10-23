const LABELS: Record<string, string> = {
  department: "Dept",
  semester: "Sem",
  course_code: "Course",
  program: "Programme",
};

/** Shows the slots the backend is remembering, so multi-turn context is visible. */
export default function ContextBar({ slots }: { slots: Record<string, string | number> }) {
  const entries = Object.entries(slots);
  if (entries.length === 0) return null;
  return (
    <div className="context" aria-label="Conversation context">
      <span className="context-label">Remembering</span>
      {entries.map(([key, value]) => (
        <span key={key} className="tag">
          {LABELS[key] ?? key}: {value}
        </span>
      ))}
    </div>
  );
}
