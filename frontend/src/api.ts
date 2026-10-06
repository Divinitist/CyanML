import type { QuestionDetail, QuestionSummary, TestResult } from "./types";

const API_ROOT = "http://127.0.0.1:8765/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `请求失败：${response.status}`);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return response.json() as Promise<T>;
}

export const api = {
  listQuestions: () => request<QuestionSummary[]>("/questions"),
  getQuestion: (id: string) => request<QuestionDetail>(`/questions/${id}`),
  saveSubmission: (id: string, code: string) =>
    request<void>(`/questions/${id}/submission`, {
      method: "PUT",
      body: JSON.stringify({ code }),
    }),
  resetSubmission: (id: string) =>
    request<{ code: string }>(`/questions/${id}/reset`, { method: "POST" }),
  runTests: (id: string) =>
    request<TestResult>(`/questions/${id}/test`, { method: "POST" }),
};
