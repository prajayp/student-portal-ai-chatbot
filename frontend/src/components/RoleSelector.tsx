import type { Role } from "../types";

const ROLES: { value: Role; label: string }[] = [
  { value: "student", label: "Student" },
  { value: "faculty", label: "Faculty" },
  { value: "administrator", label: "Admin" },
];

interface Props {
  value: Role;
  onChange: (role: Role) => void;
  disabled?: boolean;
}

export default function RoleSelector({ value, onChange, disabled }: Props) {
  return (
    <div className="roles" role="radiogroup" aria-label="Signed in as">
      {ROLES.map((r) => (
        <button
          key={r.value}
          type="button"
          role="radio"
          aria-checked={value === r.value}
          className={value === r.value ? "role active" : "role"}
          onClick={() => onChange(r.value)}
          disabled={disabled}
        >
          {r.label}
        </button>
      ))}
    </div>
  );
}
