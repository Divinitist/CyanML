import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, PointerEvent as ReactPointerEvent } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { acceptCompletion } from "@codemirror/autocomplete";
import { indentWithTab } from "@codemirror/commands";
import { python } from "@codemirror/lang-python";
import { Prec } from "@codemirror/state";
import { keymap } from "@codemirror/view";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import { api } from "./api";
import type { QuestionDetail, QuestionSummary, TestResult } from "./types";

const CURRENT_QUESTION_KEY = "cyanml.current-question";
const SIDEBAR_WIDTH_KEY = "cyanml.sidebar-width";
const PROBLEM_WIDTH_KEY = "cyanml.problem-width";
const EDITOR_HEIGHT_KEY = "cyanml.editor-height";
const draftKey = (id: string) => `cyanml.draft.${id}`;
const tabCompletionKeymap = Prec.highest(
  keymap.of([
    { key: "Tab", run: acceptCompletion },
    indentWithTab,
  ]),
);

function storedWidth(key: string, fallback: number): number {
  const value = Number(localStorage.getItem(key));
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

type SaveState = "idle" | "dirty" | "saving" | "saved" | "error";

export default function App() {
  const [questions, setQuestions] = useState<QuestionSummary[]>([]);
  const [currentId, setCurrentId] = useState("");
  const [question, setQuestion] = useState<QuestionDetail | null>(null);
  const [code, setCode] = useState("");
  const [savedCode, setSavedCode] = useState("");
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [result, setResult] = useState<TestResult | null>(null);
  const [showAnswer, setShowAnswer] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [sidebarWidth, setSidebarWidth] = useState(() => storedWidth(SIDEBAR_WIDTH_KEY, 224));
  const [problemWidth, setProblemWidth] = useState(() => storedWidth(PROBLEM_WIDTH_KEY, 680));
  const [editorHeight, setEditorHeight] = useState(() => storedWidth(EDITOR_HEIGHT_KEY, 460));
  const loadedId = useRef("");
  const dirty = code !== savedCode;

  useEffect(() => {
    document.body.classList.remove("resizing-panels", "resizing-rows");
    return () => document.body.classList.remove("resizing-panels", "resizing-rows");
  }, []);

  useEffect(() => {
    api
      .listQuestions()
      .then((items) => {
        setQuestions(items);
        const remembered = localStorage.getItem(CURRENT_QUESTION_KEY);
        const initial = items.find((item) => item.id === remembered)?.id ?? items[0]?.id;
        if (initial) setCurrentId(initial);
      })
      .catch((reason: Error) => setError(reason.message));
  }, []);

  useEffect(() => {
    if (!currentId) return;
    loadedId.current = "";
    setQuestion(null);
    setResult(null);
    setShowAnswer(false);
    setError("");
    api
      .getQuestion(currentId)
      .then((detail) => {
        const draft = localStorage.getItem(draftKey(currentId));
        setQuestion(detail);
        setResult(detail.last_result);
        setSavedCode(detail.submission);
        setCode(draft ?? detail.submission);
        setSaveState(draft !== null && draft !== detail.submission ? "dirty" : "idle");
        loadedId.current = currentId;
        localStorage.setItem(CURRENT_QUESTION_KEY, currentId);
      })
      .catch((reason: Error) => setError(reason.message));
  }, [currentId]);

  const save = useCallback(async () => {
    if (!currentId || loadedId.current !== currentId || code === savedCode) return;
    setSaveState("saving");
    try {
      await api.saveSubmission(currentId, code);
      setSavedCode(code);
      localStorage.removeItem(draftKey(currentId));
      setSaveState("saved");
    } catch (reason) {
      setSaveState("error");
      setError(reason instanceof Error ? reason.message : "保存失败");
      throw reason;
    }
  }, [code, currentId, savedCode]);

  useEffect(() => {
    if (!dirty || loadedId.current !== currentId) return;
    localStorage.setItem(draftKey(currentId), code);
    setSaveState("dirty");
    const timer = window.setTimeout(() => void save(), 900);
    return () => window.clearTimeout(timer);
  }, [code, currentId, dirty, save]);

  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => {
      if (!dirty) return;
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  const switchQuestion = (nextId: string) => {
    if (nextId === currentId) return;
    if (dirty && !window.confirm("当前修改尚未保存，仍要切换题目吗？")) return;
    setCurrentId(nextId);
  };

  const beginResize = (
    panel: "sidebar" | "problem",
    event: ReactPointerEvent<HTMLDivElement>,
  ) => {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = panel === "sidebar" ? sidebarWidth : problemWidth;
    const minimum = panel === "sidebar" ? 160 : 320;
    const maximum = panel === "sidebar" ? 420 : 1000;

    const move = (moveEvent: PointerEvent) => {
      const nextWidth = Math.min(maximum, Math.max(minimum, startWidth + moveEvent.clientX - startX));
      if (panel === "sidebar") setSidebarWidth(nextWidth);
      else setProblemWidth(nextWidth);
    };
    const finish = () => {
      document.body.classList.remove("resizing-panels");
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
      window.removeEventListener("blur", finish);
    };
    document.body.classList.add("resizing-panels");
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", finish, { once: true });
    window.addEventListener("pointercancel", finish, { once: true });
    window.addEventListener("blur", finish, { once: true });
  };

  const beginEditorResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const startY = event.clientY;
    const startHeight = editorHeight;
    const move = (moveEvent: PointerEvent) => {
      setEditorHeight(Math.min(760, Math.max(180, startHeight + moveEvent.clientY - startY)));
    };
    const finish = () => {
      document.body.classList.remove("resizing-rows");
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
      window.removeEventListener("pointercancel", finish);
      window.removeEventListener("blur", finish);
    };
    document.body.classList.add("resizing-rows");
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", finish, { once: true });
    window.addEventListener("pointercancel", finish, { once: true });
    window.addEventListener("blur", finish, { once: true });
  };

  useEffect(() => {
    localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth));
  }, [sidebarWidth]);

  useEffect(() => {
    localStorage.setItem(PROBLEM_WIDTH_KEY, String(problemWidth));
  }, [problemWidth]);

  useEffect(() => {
    localStorage.setItem(EDITOR_HEIGHT_KEY, String(editorHeight));
  }, [editorHeight]);

  const reset = async () => {
    if (!currentId || !window.confirm("确定用题目模板覆盖当前作答吗？")) return;
    try {
      const response = await api.resetSubmission(currentId);
      setCode(response.code);
      setSavedCode(response.code);
      localStorage.removeItem(draftKey(currentId));
      setSaveState("saved");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "重置失败");
    }
  };

  const run = async () => {
    if (!currentId) return;
    setRunning(true);
    setError("");
    try {
      await save();
      const nextResult = await api.runTests(currentId);
      setResult(nextResult);
      setQuestions((items) =>
        items.map((item) =>
          item.id === currentId ? { ...item, last_score: nextResult.score } : item,
        ),
      );
      setQuestion((current) =>
        current
          ? { ...current, last_score: nextResult.score, last_result: nextResult }
          : current,
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "测试失败");
    } finally {
      setRunning(false);
    }
  };

  const saveLabel = useMemo(
    () =>
      ({
        idle: "已同步",
        dirty: "等待自动保存",
        saving: "保存中…",
        saved: "已保存",
        error: "保存失败",
      })[saveState],
    [saveState],
  );

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">C</span>
          <div><strong>CyanML</strong><small>机器学习手撕题库</small></div>
        </div>
        <div className={`save-state ${saveState}`}>{saveLabel}</div>
      </header>

      <div
        className="workspace"
        style={{
          "--sidebar-width": `${sidebarWidth}px`,
          "--problem-width": `${problemWidth}px`,
        } as CSSProperties}
      >
        <aside className="question-list" aria-label="题目列表">
          <div className="list-heading"><span>题目</span><span>{questions.length}</span></div>
          {questions.map((item) => (
            <button
              className={item.id === currentId ? "question active" : "question"}
              key={item.id}
              onClick={() => switchQuestion(item.id)}
            >
              <span className="question-id">
                  {String(item.order ?? Number(item.id)).padStart(2, "0")}
                </span>
              <span className="question-copy">
                <span className="question-title-row">
                  <strong>{item.title}</strong>
                  {item.last_score !== null && (
                    <span
                      className={`question-score ${item.last_score === 100 ? "passed" : "failed"}`}
                      aria-label={`最近一次得分 ${item.last_score}`}
                    >
                      {item.last_score}
                    </span>
                  )}
                </span>
                <small>{item.library}</small>
              </span>
            </button>
          ))}
        </aside>

        <div
          className="resize-handle"
          data-testid="sidebar-resizer"
          role="separator"
          aria-label="调整题目列表宽度"
          aria-orientation="vertical"
          onPointerDown={(event) => beginResize("sidebar", event)}
        />

        <section className="problem-pane">
          {question ? (
            <article className="markdown">
              <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
                {question.description}
              </ReactMarkdown>
            </article>
          ) : <div className="empty-state">{error || "正在加载题目…"}</div>}
        </section>

        <div
          className="resize-handle"
          data-testid="problem-resizer"
          role="separator"
          aria-label="调整题面与代码区宽度"
          aria-orientation="vertical"
          onPointerDown={(event) => beginResize("problem", event)}
        />

        <section
          className="work-pane"
          style={{ "--editor-height": `${editorHeight}px` } as CSSProperties}
        >
          <div className="editor-heading">
            <span>{showAnswer ? "Python · 标准答案" : "Python · 我的作答"}</span>
            <div className="actions">
              <button
                className="secondary"
                onClick={() => setShowAnswer((current) => !current)}
                disabled={!question}
              >
                {showAnswer ? "返回作答" : "查看标答"}
              </button>
              <button className="secondary" onClick={() => void save()} disabled={showAnswer || !dirty}>保存</button>
              <button className="secondary" onClick={() => void reset()} disabled={showAnswer}>重置</button>
              <button className="primary" onClick={() => void run()} disabled={showAnswer || running}>
                {running ? "测试中…" : "运行 100 组测试"}
              </button>
            </div>
          </div>
          <div className="editor-wrap">
            <CodeMirror
              value={showAnswer ? question?.answer ?? "" : code}
              height="100%"
              extensions={[python(), tabCompletionKeymap]}
              onChange={setCode}
              theme="dark"
              editable={!showAnswer}
              readOnly={showAnswer}
              indentWithTab={false}
              basicSetup={{
                foldGutter: true,
                lineNumbers: true,
                highlightActiveLine: true,
                tabSize: 4,
              }}
            />
          </div>
          <div
            className="resize-handle horizontal"
            data-testid="editor-resizer"
            role="separator"
            aria-label="调整代码框与测试结果高度"
            aria-orientation="horizontal"
            onPointerDown={beginEditorResize}
          />
          <TestPanel result={result} />
        </section>
      </div>
      {error && <div className="toast" role="alert">{error}</div>}
    </main>
  );
}

function TestPanel({ result }: { result: TestResult | null }) {
  const [selectedCaseId, setSelectedCaseId] = useState<number | null>(null);

  useEffect(() => {
    if (!result) {
      setSelectedCaseId(null);
      return;
    }
    setSelectedCaseId(result.cases.find((item) => !item.passed)?.id ?? result.cases[0]?.id ?? null);
  }, [result]);

  if (!result) return <section className="test-panel empty">运行测试后在此查看逐例结果。</section>;
  const selectedCase = result.cases.find((item) => item.id === selectedCaseId) ?? result.cases[0];
  return (
    <section className="test-panel">
      <div className="result-summary">
        <strong className={result.failed === 0 ? "pass" : "fail"}>
          {result.passed} / {result.total} 通过
        </strong>
        <span>{result.duration_ms?.toFixed(1)} ms</span>
      </div>
      {result.stdout && (
        <section className="stdout-panel">
          <h4>全局标准输出</h4>
          <pre>{result.stdout}</pre>
        </section>
      )}
      {result.error && <pre className="global-error">{result.error}</pre>}
      <div className="case-tabs" aria-label="测试用例">
        {result.cases.map((item) => (
          <button
            className={`case-tab ${item.passed ? "passed" : "failed"} ${item.id === selectedCase?.id ? "active" : ""}`}
            key={item.id}
            onClick={() => setSelectedCaseId(item.id)}
          >
            <span>{item.passed ? "✓" : "×"}</span>
            用例 {item.id}
          </button>
        ))}
      </div>
      {selectedCase && (
        <div className="case-detail">
          <div className="case-title">
            <span className={selectedCase.passed ? "status-pass" : "status-fail"}>
              {selectedCase.passed ? "通过" : "未通过"}
            </span>
            <strong>#{selectedCase.id} {selectedCase.name}</strong>
          </div>
          {selectedCase.stdout && (
            <section className="stdout-panel case-stdout">
              <h4>本用例标准输出</h4>
              <pre>{selectedCase.stdout}</pre>
            </section>
          )}
          <ResultValue label="输入" value={selectedCase.input} wide />
          <div className="output-comparison">
            <ResultValue label="实际输出" value={selectedCase.actual} />
            <ResultValue label="预期输出" value={selectedCase.expected} />
          </div>
          {selectedCase.error && (
            <div className="case-error">
              <span>错误信息</span>
              <code>{selectedCase.error}</code>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function ResultValue({ label, value, wide = false }: { label: string; value: unknown; wide?: boolean }) {
  return (
    <section className={`result-value ${wide ? "wide" : ""}`}>
      <h4>{label}</h4>
      <div className="value-surface"><PrettyValue value={value} /></div>
    </section>
  );
}

function PrettyValue({ value }: { value: unknown }) {
  if (value === null || value === undefined) return <span className="value-muted">无输出</span>;
  if (typeof value === "string") return <code className="scalar-value">{value}</code>;
  if (typeof value === "number") return <code className="scalar-value">{formatNumber(value)}</code>;
  if (typeof value === "boolean") return <code className="scalar-value">{value ? "true" : "false"}</code>;
  if (Array.isArray(value)) {
    const simple = value.every((item) => ["string", "number", "boolean"].includes(typeof item) || item === null);
    if (simple) return <code className="array-value">[{value.map(formatScalar).join(", ")}]</code>;
    return <div className="value-list">{value.map((item, index) => <PrettyValue value={item} key={index} />)}</div>;
  }
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    if (Array.isArray(record.shape) && Array.isArray(record.values)) {
      return (
        <div className="tensor-value">
          <span className="shape-badge">shape {record.shape.join(" × ")}</span>
          <code>[{record.values.map(formatScalar).join(", ")}{record.truncated ? ", …" : ""}]</code>
        </div>
      );
    }
    return (
      <dl className="value-fields">
        {Object.entries(record).map(([key, item]) => (
          <div className="value-field" key={key}>
            <dt>{key}</dt>
            <dd><PrettyValue value={item} /></dd>
          </div>
        ))}
      </dl>
    );
  }
  return <code className="scalar-value">{String(value)}</code>;
}

function formatScalar(value: unknown): string {
  if (typeof value === "number") return formatNumber(value);
  if (typeof value === "string") return value;
  if (value === null) return "null";
  return String(value);
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value) || Number.isInteger(value)) return String(value);
  return Number(value.toPrecision(8)).toString();
}
