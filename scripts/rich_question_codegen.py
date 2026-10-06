from __future__ import annotations

from question_codegen import generate


def rich_generate(spec: dict) -> None:
    input_lines = "\n".join(
        f"- `{name}`：{description}" for name, description in spec["inputs"]
    )
    args_lines = "\n".join(
        f"    {name}: {description}" for name, description in spec["inputs"]
    )
    spec = dict(spec)
    spec["problem"] = f'''# {spec["title"]}

## 背景

{spec["background"]} 本题只使用 NumPy 展开核心计算，重点是理解数据流、张量形状和数值细节。

## 需求

实现 `{spec["function"]}`。{spec["requirement"]} 所有输入均符合下述约束，不要求处理非法参数，也不得修改调用者传入的数组。

## 输入

{input_lines}

除特别说明外，参与计算的数组和浮点常量均使用 `float64`。

## 输出

{spec["output"]}

## 数学定义

{spec["math"]}

## 测试约定

使用固定随机种子生成 100 组输入输出，覆盖{spec["coverage"]}。本题误差标准为 `rtol={spec["rtol"]}`、`atol={spec["atol"]}`。测试仅调用公开函数，答案所需的实现提示应写在答案代码注释中，而不是题面中。
'''
    spec["docstring"] = f'''"""
Args:
{args_lines}

Returns:
    {spec["output"]}
"""'''
    generate(spec)


def rich_generate_all(specs: list[dict]) -> None:
    for spec in specs:
        rich_generate(spec)
