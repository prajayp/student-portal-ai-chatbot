interface Props {
  suggestions: string[];
  onPick: (text: string) => void;
  disabled?: boolean;
}

export default function SuggestionChips({ suggestions, onPick, disabled }: Props) {
  if (suggestions.length === 0) return null;
  return (
    <div className="chips" aria-label="Suggested questions">
      {suggestions.map((s) => (
        <button key={s} type="button" className="chip" onClick={() => onPick(s)} disabled={disabled}>
          {s}
        </button>
      ))}
    </div>
  );
}
