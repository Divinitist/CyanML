from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent, indent


ROOT = Path(__file__).resolve().parents[1]


def generate(spec: dict[str, str]) -> None:
    """Write one self-contained CyanML exercise from a compact specification."""
    folder = ROOT / "questions" / f"{spec['id']}_{spec['slug']}"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": spec["id"],
        "slug": spec["slug"].replace("_", "-"),
        "title": spec["title"],
        "entrypoint": spec["function"],
        "library": "NumPy",
        "difficulty": spec.get("difficulty", "中等"),
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (folder / "problem.md").write_text(
        dedent(spec["problem"]).strip() + "\n", encoding="utf-8"
    )

    signature = dedent(spec["signature"]).strip()
    docstring = indent(dedent(spec["docstring"]).strip(), "    ")
    base = "import numpy as np\n\n\n" + signature + "\n"
    starter = base + docstring + "\n    pass\n"
    (folder / "template.py").write_text(starter, encoding="utf-8")
    (folder / "submission.py").write_text(starter, encoding="utf-8")
    answer = base + docstring + "\n" + indent(dedent(spec["answer"]).strip(), "    ") + "\n"
    (folder / "answer.py").write_text(answer, encoding="utf-8")

    cases = indent(dedent(spec["cases"]).strip(), "        ").lstrip()
    test = dedent(
        f'''\
        from __future__ import annotations

        import json
        import sys
        from pathlib import Path

        import numpy as np

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from test_support import run_cases

        from answer import {spec["function"]} as reference
        from submission import {spec["function"]}


        {cases}


        def run():
            return run_cases(
                build_cases(),
                {spec["function"]},
                reference,
                rtol={spec.get("rtol", "1e-8")},
                atol={spec.get("atol", "1e-10")},
                mismatch_message="输出与参考实现不一致。",
            )


        if __name__ == "__main__":
            print("CYANML_RESULT=" + json.dumps(run(), ensure_ascii=False, allow_nan=False))
        '''
    )
    (folder / "test.py").write_text(test, encoding="utf-8")


def generate_all(specs: list[dict[str, str]]) -> None:
    for spec in specs:
        generate(spec)
