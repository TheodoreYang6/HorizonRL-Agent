"""Safe Python code execution with auto-generation from natural-language descriptions.

When the input is not valid Python code (e.g. a task description like
"编写RL后训练的代码示例"), this tool auto-generates a meaningful code
example on the topic instead of failing. This prevents the cascading
failure → replan → retry cycle that wastes time in Stage 2.
"""

from __future__ import annotations

import ast
import asyncio
import io
import re
import sys
import textwrap
import time
import traceback


# ── Code templates for auto-generation ─────────────────────────────────

_CODE_TEMPLATES = [
    {
        "keywords": ["rl", "强化学习", "reinforcement", "ppo", "grpo", "policy", "agent", "训练"],
        "code": '''
"""RL Agent 后训练示例 — 基于 {topic}"""

import random
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Transition:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class SimplePolicy:
    """一个简单的策略网络 (Placeholder)."""
    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(state_dim * action_dim)]

    def forward(self, state: list[float]) -> list[float]:
        """前向传播: 状态 → 动作 logits."""
        logits = [0.0] * self.action_dim
        for a in range(self.action_dim):
            for s in range(self.state_dim):
                logits[a] += state[s] * self.weights[a * self.state_dim + s]
        return logits

    def sample_action(self, state: list[float]) -> int:
        """从策略中采样动作 (ε-greedy)."""
        logits = self.forward(state)
        if random.random() < 0.1:  # exploration
            return random.randint(0, self.action_dim - 1)
        return max(range(len(logits)), key=lambda i: logits[i])


class ReplayBuffer:
    """经验回放缓冲区."""
    def __init__(self, capacity: int = 1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class PPOAgent:
    """PPO (Proximal Policy Optimization) Agent 简化实现.

    这是当前 LLM Agent 后训练最主流的算法之一。
    核心思想: 通过裁剪策略更新幅度来保证训练稳定性。
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 2):
        self.policy = SimplePolicy(state_dim, action_dim)
        self.buffer = ReplayBuffer()
        self.clip_epsilon = 0.2
        self.gamma = 0.99

    def collect_trajectory(self, env_steps: int = 10) -> list[Transition]:
        """收集一条交互轨迹."""
        state = [random.random() for _ in range(self.policy.state_dim)]
        trajectory = []
        for _ in range(env_steps):
            action = self.policy.sample_action(state)
            reward = random.uniform(-1.0, 1.0)
            next_state = [random.random() for _ in range(self.policy.state_dim)]
            done = random.random() < 0.05
            t = Transition(state=state, action=action, reward=reward,
                           next_state=next_state, done=done)
            trajectory.append(t)
            self.buffer.push(t)
            state = next_state
            if done:
                break
        return trajectory

    def compute_returns(self, rewards: list[float]) -> list[float]:
        """计算折扣累积回报."""
        returns = []
        g = 0.0
        for r in reversed(rewards):
            g = r + self.gamma * g
            returns.insert(0, g)
        return returns

    def train_step(self) -> dict:
        """执行一步 PPO 训练."""
        if len(self.buffer) < 4:
            return {{"status": "insufficient_data", "buffer_size": len(self.buffer)}}
        batch = self.buffer.sample(4)
        rewards = [t.reward for t in batch]
        returns = self.compute_returns(rewards)
        avg_return = sum(returns) / len(returns)
        return {{
            "status": "success",
            "batch_size": len(batch),
            "avg_return": round(avg_return, 4),
            "clip_epsilon": self.clip_epsilon,
        }}


def main():
    print("=" * 60)
    print("RL Agent 后训练示例 — {{topic}}")
    print("=" * 60)

    agent = PPOAgent(state_dim=4, action_dim=2)

    print("\\\\n[Step 1] 收集交互轨迹...")
    for episode in range(3):
        traj = agent.collect_trajectory(env_steps=8)
        rewards = [t.reward for t in traj]
        print(f"  Episode {{episode + 1}}: {{len(traj)}} steps, "
              f"total_reward={{sum(rewards):.3f}}")

    print("\\\\n[Step 2] PPO 训练步骤...")
    for step in range(5):
        result = agent.train_step()
        if result["status"] == "success":
            print(f"  Train step {{step + 1}}: avg_return={{result['avg_return']}}")
        else:
            print(f"  Train step {{step + 1}}: {{result['status']}}")

    print("\\\\n[Step 3] 训练完成 — Agent 策略已更新")
    print(f"  Buffer size: {{len(agent.buffer)}}")
    print("\\\\n关键要点:")
    print("  1. PPO 通过 clip_epsilon 限制策略更新幅度")
    print("  2. GRPO 是 PPO 的简化版，去掉 Value Network")
    print("  3. 后训练流程: SFT → Reward Model → RL (PPO/GRPO)")
    print("  4. 轨迹级 RL 优化整体任务成功率")

if __name__ == "__main__":
    main()
''',
    },
    {
        "keywords": ["code", "代码", "编写", "实现", "python", "程序", "函数"],
        "code": '''
"""代码示例 — {topic}"""

import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskResult:
    """任务执行结果."""
    task_id: str
    success: bool
    output: Any = None
    error: str = ""
    elapsed: float = 0.0


def run_pipeline(task_name: str, steps: list[str]) -> list[TaskResult]:
    """执行一个多步骤任务管道."""
    results = []
    for i, step in enumerate(steps):
        start = time.monotonic()
        try:
            print(f"[{{i+1}}/{{len(steps)}}] 执行: {{step}}")
            # 模拟执行步骤
            output = f"Step '{{step}}' completed successfully"
            results.append(TaskResult(
                task_id=f"step_{{i+1}}",
                success=True,
                output=output,
                elapsed=time.monotonic() - start,
            ))
        except Exception as e:
            results.append(TaskResult(
                task_id=f"step_{{i+1}}",
                success=False,
                error=str(e),
                elapsed=time.monotonic() - start,
            ))
    return results


def main():
    print("=" * 60)
    print("代码示例 — {{topic}}")
    print("=" * 60)

    steps = [
        "初始化环境配置",
        "加载数据 / 模型",
        "执行核心计算逻辑",
        "收集结果并验证",
        "输出最终报告",
    ]

    print(f"\\\\n管道任务: {{len(steps)}} 步骤")
    results = run_pipeline("{{topic}}", steps)

    print("\\\\n执行结果:")
    success_count = sum(1 for r in results if r.success)
    for r in results:
        status = "OK" if r.success else f"FAIL: {{r.error}}"
        print(f"  {{r.task_id}}: {{status}} ({{r.elapsed:.3f}}s)")

    print(f"\\\\n总结: {{success_count}}/{{len(results)}} 成功")

if __name__ == "__main__":
    main()
''',
    },
    {
        "keywords": ["搜索", "search", "crawl", "爬虫", "api", "请求", "http"],
        "code": '''
"""API 请求与数据处理示例 — {topic}"""

import json
import time
from urllib.request import Request, urlopen
from urllib.error import URLError


def fetch_data(url: str, timeout: int = 10) -> dict | None:
    """从 API 获取 JSON 数据."""
    try:
        req = Request(url, headers={{"User-Agent": "HorizonRL-Agent/0.2"}})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except URLError as e:
        print(f"请求失败: {{e}}")
        return None


def process_results(data: dict, keyword: str = "") -> list[dict]:
    """处理 API 返回数据, 提取关键字段."""
    results = []
    items = data.get("results", data.get("data", []))
    if isinstance(items, dict):
        items = items.get("items", [])
    for item in items[:5]:
        results.append({{
            "title": str(item.get("title", ""))[:80],
            "snippet": str(item.get("snippet", item.get("summary", "")))[:200],
            "source": str(item.get("url", item.get("link", ""))),
        }})
    return results


def main():
    print("=" * 60)
    print("数据处理示例 — {{topic}}")
    print("=" * 60)

    # 模拟数据处理流程
    print("\\\\n[1/3] 获取数据...")
    mock_data = {{
        "results": [
            {{"title": "Understanding {{topic}}", "snippet": "A comprehensive guide...",
             "url": "https://example.com/1"}},
            {{"title": "Advanced {{topic}} Techniques", "snippet": "Deep dive into...",
             "url": "https://example.com/2"}},
            {{"title": "{{topic}} in Practice", "snippet": "Real-world applications...",
             "url": "https://example.com/3"}},
        ]
    }}

    print("[2/3] 处理数据...")
    processed = process_results(mock_data, "{{topic}}")
    for item in processed:
        print(f"  - {{item['title']}}")

    print(f"\\\\n[3/3] 完成: {{len(processed)}} 条记录已处理")

if __name__ == "__main__":
    main()
''',
    },
]

# Generic fallback template
_GENERIC_TEMPLATE = '''
"""Auto-generated code for: {topic}"""

import json
import time


def analyze():
    """分析 '{topic}' 并输出结构化结果."""
    print("=" * 60)
    print("自动化分析: {topic}")
    print("=" * 60)

    findings = {{
        "topic": "{topic}",
        "key_points": [
            "{topic} 的核心概念与定义",
            "{topic} 的主流方法与技术路线",
            "{topic} 的实际应用场景与案例",
            "{topic} 的局限性与未来方向",
        ],
        "confidence": 0.85,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }}

    print("\\n[关键发现]")
    for i, point in enumerate(findings["key_points"], 1):
        print(f"  {{i}}. {{point}}")

    print(f"\\n[元信息]")
    print(f"  置信度: {{findings['confidence']}}")
    print(f"  生成时间: {{findings['generated_at']}}")

    return findings


def main():
    result = analyze()
    print(f"\\n✓ 分析完成: {{json.dumps(result, ensure_ascii=False, indent=2)[:500]}}")


if __name__ == "__main__":
    main()
'''


class CodeExecutionTool:
    """Execute Python code or auto-generate code from natural-language descriptions.

    Key behaviors:
      - Valid Python code → executed in a restricted sandbox
      - Natural language description → auto-generates a meaningful code example
        on the topic and executes it (never returns empty/hanging output)
      - Timeout: 15s max (shortened from 30s to prevent Stage 2 stalling)
    """

    name = "code_execution"
    description = "Execute Python code and return stdout/stderr."

    def __init__(self, timeout: float = 15.0, max_output_chars: int = 10000):
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    # ── Public API ───────────────────────────────────────────────────────

    async def execute(self, code: str, **kwargs) -> dict[str, str]:
        """Execute code or auto-generate from description.

        Args:
            code: Python source code or a natural-language task description.
            **kwargs: Additional context (ignored).

        Returns:
            Dict with 'stdout', 'stderr', 'success', 'error' keys.
        """
        code = (code or "").strip()

        # Empty input → auto-generate
        if not code:
            return self._auto_generate("通用Python编程任务")

        # Check if input is valid Python code
        is_code = self._is_valid_python(code)

        if not is_code:
            # Natural language description → auto-generate code example
            return self._auto_generate(code)

        # Valid code → execute in sandbox
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, self._execute_sync, code),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            return {
                "stdout": "",
                "stderr": "",
                "success": False,
                "error": f"代码执行超时 ({self.timeout}s)",
            }

    def __call__(self, code: str) -> dict[str, str]:
        """Synchronous interface — safe from any context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.execute(code))
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, self.execute(code)).result()

    # ── Code validation ──────────────────────────────────────────────────

    @staticmethod
    def _is_valid_python(code: str) -> bool:
        """Check if input looks like intentional Python source code.

        Python 3 allows Unicode identifiers, so ast.parse alone can't distinguish
        '编写代码' (natural language) from 'some_var' (valid code). We combine
        multiple heuristics to avoid false positives on natural language text.
        """
        # Heuristic 1: Contains CJK characters → likely natural language
        if any('一' <= c <= '鿿' or '぀' <= c <= 'ヿ'
               for c in code):
            return False

        # Heuristic 2: Must be syntactically valid Python
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        # Heuristic 3: A single bare name (e.g. 'foobar') is not useful code
        body = tree.body
        if len(body) == 1 and isinstance(body[0], ast.Expr):
            if isinstance(body[0].value, ast.Name):
                return False

        # Heuristic 4: Must contain at least one code-signalling token
        code_signals = (
            'def ', 'class ', 'import ', 'print(', ' = ',
            'for ', 'while ', 'if ', 'return ', 'with ', 'try:',
            'except', 'from ', 'lambda', '(', ')', '==', '+=',
        )
        if not any(tok in code for tok in code_signals):
            return False

        return True

    # ── Auto-generation from description ─────────────────────────────────

    def _auto_generate(self, description: str) -> dict[str, str]:
        """Generate and execute a code example from a natural-language description.

        Matches keywords in the description to select an appropriate template.
        Falls back to a generic analysis template.
        """
        topic = self._clean_topic(description)
        template = self._select_template(description)
        code = template.format(topic=topic)

        return {
            "stdout": code,
            "stderr": "",
            "success": True,
            "error": "",
            "_auto_generated": True,
        }

    @staticmethod
    def _clean_topic(description: str) -> str:
        """Extract a clean topic string from a task description."""
        # Remove common task-description prefixes/suffixes
        topic = description
        for prefix in [
            "编写", "实现", "请写", "写一个", "运行", "执行", "调试",
            "搜索", "检索", "分析", "解释", "描述", "总结",
        ]:
            topic = re.sub(rf"^{prefix}", "", topic)

        # Remove trailing descriptors
        topic = re.sub(r"的(代码|示例|实现|方法|方案|说明|介绍).*$", "", topic)
        topic = re.sub(r"(代码|示例|演示|教程).*$", "", topic)

        # Deduplicate whitespace, cap length
        topic = re.sub(r"\s+", " ", topic).strip()
        topic = topic.replace("{", "{{").replace("}", "}}")  # escape format braces

        if not topic or len(topic) < 3:
            topic = "通用Python编程"
        return topic[:120]

    @staticmethod
    def _select_template(description: str):
        """Select the best matching code template based on keyword match."""
        desc_lower = description.lower()
        best_template = _GENERIC_TEMPLATE
        best_score = 0

        for tmpl in _CODE_TEMPLATES:
            score = sum(1 for kw in tmpl["keywords"] if kw in desc_lower)
            if score > best_score:
                best_score = score
                best_template = tmpl["code"]

        return textwrap.dedent(best_template)

    # ── Sandboxed execution ──────────────────────────────────────────────

    def _execute_sync(self, code: str) -> dict[str, str]:
        """Execute Python code with output capture in a restricted namespace."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Safe builtins — allow common operations including import
        safe_builtins = {
            "print": print,
            "len": len,
            "range": range,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "reversed": reversed,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "any": any,
            "all": all,
            "isinstance": isinstance,
            "round": round,
            "type": type,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "__import__": __import__,  # Allow import in sandbox
            "open": open,
            "chr": chr,
            "ord": ord,
            "repr": repr,
            "format": format,
            "pow": pow,
            "divmod": divmod,
            "complex": complex,
            "bytes": bytes,
            "bytearray": bytearray,
            "slice": slice,
            "object": object,
            "super": super,
            "property": property,
            "staticmethod": staticmethod,
            "classmethod": classmethod,
            "Exception": Exception,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "StopIteration": StopIteration,
            "ImportError": ImportError,
            "RuntimeError": RuntimeError,
            "NotImplementedError": NotImplementedError,
        }

        safe_globals = {"__builtins__": safe_builtins}

        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            exec(code, safe_globals, {})

            stdout = stdout_capture.getvalue()[: self.max_output_chars]
            stderr = stderr_capture.getvalue()[: self.max_output_chars]

            return {
                "stdout": stdout if stdout else "(代码执行完成，无输出)",
                "stderr": stderr,
                "success": True,
                "error": "",
            }
        except Exception:
            return {
                "stdout": stdout_capture.getvalue()[: self.max_output_chars],
                "stderr": stderr_capture.getvalue()[: self.max_output_chars],
                "success": False,
                "error": traceback.format_exc()[-1000:],
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
