import os

import jax

from alpa.global_env import is_worker

trace_enabled = os.environ.get("ALPA_TRACE_ENABLED", "False") == "True"
trace_log_base_dir = os.environ.get("ALPA_TRACE_LOG_BASE_DIR", "")

def worker_start_tracing(name):
  if not trace_enabled or not is_worker:
    print(f"worker_start_tracing: trace_enabled={trace_enabled}, is_worker={is_worker}")
    return
  print(f"worker_start_tracing: {trace_log_base_dir}/{name}")
  jax.profiler.start_trace(f"{trace_log_base_dir}/{name}")

def worker_stop_tracing():
  if not trace_enabled or not is_worker:
    print(f"worker_stop_tracing: trace_enabled={trace_enabled}, is_worker={is_worker}")
    return
  jax.profiler.stop_trace()
