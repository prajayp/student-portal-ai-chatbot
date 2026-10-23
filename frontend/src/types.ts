export type Role = "student" | "faculty" | "administrator";

export interface ChatResponse {
  reply: string;
  intent: string;
  confidence: number;
  suggestions: string[];
  slots: Record<string, string | number>;
  resolved_by: "classifier" | "follow_up" | "clarification" | "fallback" | "restricted";
}

export interface Message {
  id: string;
  from: "user" | "bot";
  text: string;
  meta?: Omit<ChatResponse, "reply" | "suggestions">;
  error?: boolean;
}
