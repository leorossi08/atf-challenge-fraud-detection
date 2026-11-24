# Runs ON the Colab VM (via `colab exec -f`). Prints new training-log content + run state.
#   --env OFFSET=<bytes>  -> print everything after that byte offset (used by the runner script)
#   --env N=<lines>       -> print the last N lines (used for manual watching; default 25)
import os
import pathlib

log = pathlib.Path("/content/train.log")
exit_file = pathlib.Path("/content/train.exit")
data = log.read_bytes() if log.exists() else b""
offset = int(os.environ.get("OFFSET", "-1"))
if offset >= 0:
    print(data[offset:].decode(errors="replace"), end="")
else:
    lines = data.decode(errors="replace").splitlines()
    print("\n".join(lines[-int(os.environ.get("N", "25")):]))
print(f"__OFFSET__={len(data)}")
print("__EXIT__=" + (exit_file.read_text().strip() if exit_file.exists() else "RUNNING"))
