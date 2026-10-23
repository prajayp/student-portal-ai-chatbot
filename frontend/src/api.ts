import type { ChatResponse, Role } from "./types";

export async function sendMessage(sessionId: string, role: Role, message: string): Promise<ChatResponse> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, role, message }),
  });
  if (!res.ok) {
    throw new Error(`Server responded with ${res.status}`);
  }
  return res.json();
}

/** crypto.randomUUID is only available in secure contexts (not on a phone hitting a LAN IP). */
export function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
