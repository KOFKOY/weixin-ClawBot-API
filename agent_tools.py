"""智能体工具模块：封装记忆管理与定时任务管理能力。"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
import uuid
from typing import Callable

from agent_scheduler import CronScheduler


class MemoryTool:
    """记忆工具：先写入内存，再落盘到文本文件。"""

    tool_name = "memory"

    def __init__(self, memory_file: str = "agent_memory.txt"):
        """初始化内存存储结构与持久化目标文件路径。"""
        self.memory_file = memory_file
        self._lock = threading.Lock()
        self._records: list[dict] = []
        self._latest_by_key: dict[str, dict] = {}
        self._load_from_text()

    def _normalize_text(self, text: str) -> str:
        """归一化文本，便于做宽松匹配。"""
        text = (text or "").lower()
        return re.sub(r"[\s\W_]+", "", text, flags=re.U)

    def _load_from_text(self):
        """启动时加载历史文本记忆，确保重启后仍可查询。"""
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict) and row.get("key"):
                        self._records.append(row)
                        self._latest_by_key[row["key"]] = row
        except FileNotFoundError:
            return
        except Exception:
            return

    def save_memory(
        self,
        key: str,
        value: str,
        source: str = "user",
        tags: list[str] | None = None,
        aliases: list[str] | None = None,
    ) -> dict:
        """保存一条记忆到内存，并立即同步写入文本文件。"""
        if not key:
            raise ValueError("key 不能为空")
        record = {
            "id": uuid.uuid4().hex[:8],
            "key": key,
            "value": value,
            "source": source,
            "tags": tags or [],
            "aliases": aliases or [],
            "created_at": int(time.time()),
        }
        with self._lock:
            self._records.append(record)
            self._latest_by_key[key] = record
            self.flush_to_text()
        return record

    def read_memory(self, key: str) -> dict:
        """按 key 或别名读取最新记忆，不存在时返回空对象。"""
        with self._lock:
            row = self._latest_by_key.get(key)
            if row:
                return dict(row)
            for item in reversed(self._records):
                aliases = item.get("aliases") or []
                if key in aliases:
                    return dict(item)
            return {}

    def search_memory(self, query: str, limit: int = 5) -> list[dict]:
        """按关键词检索记忆，适合“XX 是多少”这类自然语言查询。"""
        q = (query or "").strip().lower()
        if not q:
            return []
        q_norm = self._normalize_text(q)
        hits: list[dict] = []
        with self._lock:
            for item in self._records:
                key_text = str(item.get("key", "")).lower()
                value_text = str(item.get("value", "")).lower()
                tags_text = " ".join(item.get("tags") or []).lower()
                aliases_text = " ".join(item.get("aliases") or []).lower()
                key_norm = self._normalize_text(key_text)
                value_norm = self._normalize_text(value_text)
                aliases_norm = self._normalize_text(aliases_text)
                score = 0
                if q in key_text:
                    score += 4
                if q in aliases_text:
                    score += 3
                if q in tags_text:
                    score += 2
                if q in value_text:
                    score += 1
                if q_norm and q_norm in key_norm:
                    score += 4
                if q_norm and q_norm in aliases_norm:
                    score += 3
                if q_norm and q_norm in value_norm:
                    score += 1

                # 字符重叠兜底：适配“我QQ邮箱的密码是多少”与“QQ邮箱密码”这类表达差异。
                if q_norm and key_norm:
                    overlap = len(set(q_norm) & set(key_norm))
                    if overlap >= 2:
                        score += overlap
                # 兜底：拆词匹配，提升“QQ邮箱 密码”这类查询命中率。
                for token in q.split():
                    if token and token in key_text:
                        score += 2
                    if token and token in aliases_text:
                        score += 1
                    if token and token in value_text:
                        score += 1
                if score > 0:
                    data = dict(item)
                    data["match_score"] = score
                    hits.append(data)
        hits.sort(key=lambda x: (x.get("match_score", 0), x.get("created_at", 0)), reverse=True)
        return hits[: max(1, limit)]

    def list_memory(self, limit: int = 20) -> list[dict]:
        """按时间倒序查看记忆列表。"""
        with self._lock:
            data = list(self._records[-max(1, limit):])
        data.reverse()
        return data

    def flush_to_text(self):
        """将当前内存中的全部记忆写入文本文件（JSON Lines）。"""
        with open(self.memory_file, "w", encoding="utf-8") as f:
            for row in self._records:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def memory_prompt_snippet(self, limit: int = 10) -> str:
        """生成可直接拼入系统提示词的记忆摘要片段。"""
        rows = self.list_memory(limit=limit)
        if not rows:
            return "暂无历史记忆"
        lines = []
        for row in rows:
            lines.append(f"- {row['key']}: {row['value']} (source={row['source']})")
        return "\n".join(lines)


class SchedulerTool:
    """任务工具：创建 cron 任务，在触发时调用 AI 生成通知文案。"""

    tool_name = "schedule"

    def __init__(
        self,
        scheduler: CronScheduler,
        ai_requester: Callable[[str, str | None], str],
        notification_queue: queue.Queue,
    ):
        """注入调度器、AI 调用器和通知队列。"""
        self.scheduler = scheduler
        self.ai_requester = ai_requester
        self.notification_queue = notification_queue
        self._tasks: dict[str, dict] = {}

    def create_cron_task(self, cron_expr: str, remind_text: str, user_id: str = "", task_name: str = "") -> dict:
        """创建或覆盖 cron 定时提醒任务。"""
        task_id = task_name.strip() or f"cron_{uuid.uuid4().hex[:8]}"
        task = {
            "task_id": task_id,
            "cron_expr": cron_expr,
            "remind_text": remind_text,
            "user_id": user_id,
            "created_at": int(time.time()),
        }
        self.scheduler.add_cron_job(task_id=task_id, cron_expr=cron_expr, func=self._execute_task, kwargs={"task_id": task_id})
        self._tasks[task_id] = task
        return task

    def remove_cron_task(self, task_id: str) -> dict:
        """删除指定 cron 任务。"""
        removed = self.scheduler.remove_job(task_id)
        if task_id in self._tasks:
            del self._tasks[task_id]
        return {"task_id": task_id, "removed": bool(removed)}

    def list_cron_tasks(self) -> list[dict]:
        """查看已创建的 cron 任务列表。"""
        return list(self._tasks.values())

    def _execute_task(self, task_id: str):
        """执行定时任务：调用 AI 生成通知文案并写入发送队列。"""
        task = self._tasks.get(task_id)
        if not task:
            return
        base_text = task["remind_text"]
        prompt = "你是提醒助手，请把提醒改写成一句简短、友好的微信通知。"
        try:
            ai_text = self.ai_requester(base_text, prompt)
        except Exception:
            ai_text = base_text
        message = ai_text.strip() if isinstance(ai_text, str) and ai_text.strip() else base_text
        self.notification_queue.put(
            {
                "task_id": task_id,
                "user_id": task.get("user_id", ""),
                "message": message,
                "created_at": int(time.time()),
            }
        )
