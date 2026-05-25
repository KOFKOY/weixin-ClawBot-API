"""定时任务模块：提供基于 cron 的任务注册、删除与查询能力。"""

from __future__ import annotations

import configparser
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger


@dataclass
class CronTaskInfo:
    """cron 任务信息模型，用于对外展示任务元数据。"""

    task_id: str
    cron_expr: str
    next_run_time: str


class CronScheduler:
    """cron 调度器封装，统一管理任务生命周期。"""

    def __init__(self, timezone: str = "Asia/Shanghai", persistence_file: str = "agent_scheduler.ini"):
        """初始化调度器（默认上海时区），但不立即重复启动。"""
        self.timezone = timezone
        self._scheduler = BackgroundScheduler(timezone=timezone)
        self._started = False
        self.persistence_file = Path(persistence_file)
        self._persist_lock = threading.Lock()

    def start(self):
        """启动调度器，重复调用时保持幂等。"""
        if self._started:
            return
        self._scheduler.start()
        self._started = True

    def shutdown(self):
        """关闭调度器，避免进程退出时残留后台线程。"""
        if not self._started:
            return
        self._scheduler.shutdown(wait=False)
        self._started = False

    def add_cron_job(self, task_id: str, cron_expr: str, func: Callable[..., Any], kwargs: dict | None = None):
        """注册或覆盖一个 cron 任务，表达式使用标准 crontab 五段格式。"""
        trigger = CronTrigger.from_crontab(cron_expr, timezone=self.timezone)
        self._scheduler.add_job(
            func=func,
            trigger=trigger,
            kwargs=kwargs or {},
            id=task_id,
            replace_existing=True,
            coalesce=True,
            misfire_grace_time=60,
        )

    def remove_job(self, task_id: str) -> bool:
        """按任务 ID 删除任务；不存在时返回 False。"""
        job = self._scheduler.get_job(task_id)
        if not job:
            return False
        self._scheduler.remove_job(task_id)
        return True

    def save_task(self, task: dict[str, Any]):
        """将任务配置写入 ini，供服务重启后恢复。"""
        task_id = str(task.get("task_id", "")).strip()
        cron_expr = str(task.get("cron_expr", "")).strip()
        if not task_id:
            raise ValueError("task_id 不能为空")
        if not cron_expr:
            raise ValueError("cron_expr 不能为空")

        section = f"task:{task_id}"
        with self._persist_lock:
            config = self._load_ini()
            config[section] = {
                "task_id": task_id,
                "cron_expr": cron_expr,
                "remind_text": str(task.get("remind_text", "")),
                "user_id": str(task.get("user_id", "")),
                "created_at": str(task.get("created_at", "")),
            }
            self._write_ini(config)

    def delete_task(self, task_id: str) -> bool:
        """从 ini 中删除任务配置。"""
        if not task_id:
            return False
        section = f"task:{task_id}"
        with self._persist_lock:
            config = self._load_ini()
            removed = config.remove_section(section)
            if removed:
                self._write_ini(config)
            return removed

    def load_tasks(self) -> list[dict[str, Any]]:
        """读取 ini 中所有任务配置。"""
        with self._persist_lock:
            config = self._load_ini()

        tasks: list[dict[str, Any]] = []
        for section in config.sections():
            if not section.startswith("task:"):
                continue
            row = config[section]
            task_id = row.get("task_id", "").strip() or section.split("task:", 1)[-1]
            cron_expr = row.get("cron_expr", "").strip()
            if not task_id or not cron_expr:
                continue
            created_at_raw = row.get("created_at", "").strip()
            try:
                created_at = int(created_at_raw) if created_at_raw else 0
            except ValueError:
                created_at = 0
            tasks.append(
                {
                    "task_id": task_id,
                    "cron_expr": cron_expr,
                    "remind_text": row.get("remind_text", ""),
                    "user_id": row.get("user_id", ""),
                    "created_at": created_at,
                }
            )
        return tasks

    def list_jobs(self) -> list[CronTaskInfo]:
        """列出当前所有定时任务及下次执行时间。"""
        result: list[CronTaskInfo] = []
        for job in self._scheduler.get_jobs():
            result.append(
                CronTaskInfo(
                    task_id=job.id,
                    cron_expr=str(job.trigger),
                    next_run_time=str(job.next_run_time) if job.next_run_time else "",
                )
            )
        return result

    def _load_ini(self) -> configparser.ConfigParser:
        """加载 ini 文件，不存在时返回空配置。"""
        config = configparser.ConfigParser(interpolation=None)
        if self.persistence_file.exists():
            config.read(self.persistence_file, encoding="utf-8")
        return config

    def _write_ini(self, config: configparser.ConfigParser):
        """原子写入 ini，避免中途写坏配置。"""
        self.persistence_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self.persistence_file.with_suffix(f"{self.persistence_file.suffix}.tmp")
        with tmp_file.open("w", encoding="utf-8") as f:
            config.write(f)
        tmp_file.replace(self.persistence_file)
