"""OpenAI 智能体封装：支持 ReAct、记忆管理与 cron 定时通知。"""

from __future__ import annotations

import inspect
import json
import queue
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from agent_scheduler import CronScheduler
from agent_tools import MemoryTool, SchedulerTool

version = "1.0.3"


def log(message, level="INFO"):
    """统一日志输出。"""
    print(f"[{level}] {message}")


@dataclass
class OpenAIConfig:
    """OpenAI 客户端配置。"""

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-5.4"
    prompt: str = "你是一个有帮助的AI助手。"
    memory_file: str = "agent_memory.ini"
    timezone: str = "Asia/Shanghai"


class OpenAIAPI:
    """OpenAI ReAct 智能体，实现工具调用、记忆管理和定时任务。"""

    def __init__(self, config: OpenAIConfig):
        """初始化基础配置，并启动工具层与定时任务模块。"""
        self.config = config
        self.api_key = config.api_key
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model
        self._notification_queue: queue.Queue = queue.Queue()

        # 工具层初始化：记忆工具 + 定时任务工具。
        self.memory_tool = MemoryTool(memory_file=config.memory_file)
        self.scheduler = CronScheduler(timezone=config.timezone)
        self.scheduler.start()
        self.scheduler_tool = SchedulerTool(
            scheduler=self.scheduler,
            ai_requester=self.request_ai_text,
            notification_queue=self._notification_queue,
        )

        # 启动时扫描工具方法与注释，构建 ReAct 提示词依据。
        self._tool_methods, self._tool_desc = self._scan_tools([self.memory_tool, self.scheduler_tool])

    def _scan_tools(self, tools: list[Any]) -> tuple[dict[str, Any], str]:
        """扫描工具类公开方法与注释，输出调用索引和描述文本。"""
        method_map: dict[str, Any] = {}
        desc_lines: list[str] = []
        for tool in tools:
            prefix = getattr(tool, "tool_name", tool.__class__.__name__.lower())
            for method_name, method in inspect.getmembers(tool, predicate=callable):
                if method_name.startswith("_"):
                    continue
                action_name = f"{prefix}.{method_name}"
                method_map[action_name] = method
                signature = str(inspect.signature(method))
                doc = inspect.getdoc(method) or "无注释"
                desc_lines.append(f"- {action_name}{signature}：{doc}")
        return method_map, "\n".join(desc_lines)

    def _build_messages(self, message: str, prompt: str, history: list[dict] | None = None) -> list[dict]:
        """构建通用 messages 结构，供模型请求复用。"""
        messages = [{"role": "system", "content": prompt}]
        if history:
            for h in history:
                role = "assistant" if h.get("attr") == "self" else "user"
                text = h.get("content", "")
                t = h.get("time", "")
                content = f"[{t}] {text}" if t else text
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})
        return messages
    

    def repeat_str(s: str) -> str:
        import re
        s = s.replace('```', '').strip()
        s = re.sub(r'}\s+{', '}{', s)
        n = len(s)
        if n % 2 == 0 and s[:n//2].strip() == s[n//2:].strip(): 
            return s[:n//2].strip()
        return s

    def request_ai(self, messages: list[dict], model: str | None = None) -> str:
        """统一 OpenAI 请求封装，便于 chat 与定时任务复用。"""
        target_model = model or self.model
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"siver-weixin_clawbot-api/{version}",
        }
        payload = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "reasoning_effort": "high",
        }
        endpoint = f"{self.base_url}/chat/completions"

        retry_delays = [2, 4, 8, 16, 32]
        max_retries = 5
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"].get("content", "")
                if content:
                    if attempt > 0:
                        log(f"OpenAIAPI 第 {attempt} 次重试成功：{content[:100]}...")
                    else:
                        content = self.repeat_str(content)
                        log(f"OpenAIAPI 返回成功：{content[:100]}...")
                    return content
                return "AI 未返回有效内容"
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = retry_delays[attempt]
                    log(f"OpenAIAPI 第 {attempt + 1} 次失败（{type(e).__name__}），{delay}s 后重试...", "WARNING")
                    time.sleep(delay)
                else:
                    log(f"OpenAIAPI 已重试 {max_retries} 次，最终失败: {last_error}", "ERROR")
        return "API接口失效，请联系管理员"

    def request_ai_text(self, text: str, prompt: str | None = None, model: str | None = None) -> str:
        """文本快捷调用入口：将文本封装成消息后直接请求模型。"""
        messages = self._build_messages(text, prompt or self.config.prompt)
        return self.request_ai(messages=messages, model=model)

    def _react_system_prompt(self, base_prompt: str) -> str:
        """构建 ReAct 系统提示词，要求模型按 JSON Schema 返回动作。"""
        memory_snippet = self.memory_tool.memory_prompt_snippet(limit=10)
        tool_names = sorted(self._tool_methods.keys())
        action_name_schema: dict[str, Any]
        if tool_names:
            action_name_schema = {"type": "string", "enum": tool_names}
        else:
            action_name_schema = {"type": "string", "minLength": 1}
        response_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "required": ["thought", "action", "final_answer"],
            "properties": {
                "thought": {
                    "type": "string",
                    "minLength": 1,
                    "description": "简洁描述本轮决策依据。",
                },
                "action": {
                    "oneOf": [
                        {"type": "null"},
                        {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "args"],
                            "properties": {
                                "name": action_name_schema,
                                "args": {"type": "object"},
                            },
                        },
                    ]
                },
                "final_answer": {
                    "type": "string",
                    "description": "最终发给用户的回复文本。若本轮仅发起工具调用可暂时为空字符串。",
                },
            },
        }
        schema_text = json.dumps(response_schema, ensure_ascii=False, indent=2)
        return (
            f"{base_prompt}\n\n"
            "你是 ReAct 智能体。根据用户输入决定是直接回答，还是调用工具。\n"
            "当前记忆摘要：\n"
            f"{memory_snippet}\n\n"
            "可用工具如下（名称、参数签名、注释）：\n"
            f"{self._tool_desc}\n\n"
            "规则：\n"
            "1) 默认直接回答；只有在确实必要时才调用工具\n"
            "2) 一轮对话最多调用一个工具，只能生成一个 action。\n"
            "3) 一轮对话最多保存一条记忆；禁止针对同一用户输入连续保存两条或以上 memory.save_memory\n"
            "4) 若用户一句话中包含多个可长期保存的信息，必须合并为一条最合适的记忆后再保存，不得拆分成多条。\n"
            "5) 仅当内容具备长期价值时才保存，例如：姓名、偏好、固定地址、证件信息、账号信息、长期计划，或用户明确要求“记住/保存/记录”。\n"
            "6) 若用户在询问“某信息是多少”，优先调用 memory.read_memory 或 memory.search_memory 查询后再回答。\n"
            "7) 需要创建定时提醒时调用 schedule.create_cron_task。\n"
            "8) 如果不需要工具，则 action 必须为 null。\n"
            "9) final_answer 必须只有一个最终版本，禁止输出备选答案、修正版、补充版或第二个结果。\n"
            "10) 严格只输出一次，且仅输出一个 JSON 对象。\n"
            "11) 禁止输出两个或多个 JSON 对象；禁止重复生成；禁止自我修正后再输出第二份 JSON。\n"
            "12) 不得在 JSON 前后添加任何解释、说明、代码块标记或多余字符。\n"
            "13) 输出前自行检查：回复内容必须能被直接解析为单个 JSON 对象；若发现自己生成了多个对象，只保留第一个合法且最合适的对象。\n"
            "14) 输出必须满足下方 JSON Schema（Draft 2020-12）。\n"
            "15) action 为 null 表示本轮不调用工具；action 非 null 时只允许一个工具调用。\n"
            "16) 每轮只返回一个 JSON 对象，且必须符合 schema。\n"
            "Response JSON Schema:\n"
            f"{schema_text}"
        )

    def _parse_json_object(self, text: str) -> dict | None:
        """从模型返回中提取 JSON 对象，兼容代码块格式。"""
        raw = text.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        match = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.S)
        if not match:
            match = re.search(r"(\{.*\})", raw, flags=re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
        return None

    def _run_tool(self, action: dict) -> str:
        """执行模型指定工具，并返回工具执行结果文本。"""
        name = action.get("name", "")
        args = action.get("args") or {}
        if name not in self._tool_methods:
            return f"工具不存在: {name}"
        if not isinstance(args, dict):
            return "工具参数必须是 JSON 对象"
        try:
            result = self._tool_methods[name](**args)
            if isinstance(result, (dict, list)):
                return json.dumps(result, ensure_ascii=False)
            return str(result)
        except Exception as e:
            return f"工具执行异常: {type(e).__name__}: {e}"

    def drain_notifications(self, limit: int = 20) -> list[dict]:
        """读取并清空通知队列，供外层微信发送逻辑消费。"""
        notices: list[dict] = []
        for _ in range(max(1, limit)):
            try:
                notices.append(self._notification_queue.get_nowait())
            except queue.Empty:
                break
        return notices

    def chat(self, message, model=None, stream=False, prompt=None, history=None):
        """ReAct 对话入口：支持工具调用、记忆写入与定时任务创建。"""
        if stream:
            log("OpenAIAPI 当前封装未启用流式响应，已按非流式请求处理", "WARN")
        target_model = model or self.model
        base_prompt = prompt or self.config.prompt

        system_prompt = self._react_system_prompt("")
        messages = self._build_messages(str(message), system_prompt, history=history)

        # 多轮 ReAct：模型可按需调用工具，但避免无限循环。
        for _ in range(3):
            raw = self.request_ai(messages=messages, model=target_model)
            parsed = self._parse_json_object(raw)
            if not parsed:
                return raw

            action = parsed.get("action")
            final_answer = str(parsed.get("final_answer", "")).strip()
            if not action:
                return final_answer or raw

            observation = self._run_tool(action)
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": f"工具执行结果：{observation}\n请继续输出相同 JSON 格式，给出最终回复。",
                }
            )
        return "工具执行完成，请继续补充细节。"
