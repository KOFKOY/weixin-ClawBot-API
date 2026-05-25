"""定时任务模块：提供基于 cron 的任务注册、删除与查询能力。"""

from __future__ import annotations

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

    def __init__(self, timezone: str = "Asia/Shanghai"):
        """初始化调度器（默认上海时区），但不立即重复启动。"""
        self.timezone = timezone
        self._scheduler = BackgroundScheduler(timezone=timezone)
        self._started = False

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
