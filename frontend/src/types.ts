export interface QuestionSummary {
  id: string;
  order?: number;
  slug: string;
  title: string;
  library: string;
  difficulty?: string;
  last_score: number | null;
}

export interface QuestionDetail extends QuestionSummary {
  description: string;
  template: string;
  submission: string;
  answer: string;
  last_result: TestResult | null;
}

export interface CaseResult {
  id: number;
  name: string;
  passed: boolean;
  input: unknown;
  expected: unknown;
  actual: unknown;
  error: string | null;
  stdout?: string;
}

export interface TestResult {
  total: number;
  passed: number;
  failed: number;
  duration_ms: number;
  cases: CaseResult[];
  score: number;
  stdout?: string;
  error?: string;
}
