"""A recorded PID must still belong to the benchmark holder before shutdown."""
import shlex
import subprocess
import sys

from sim.multisite import Remote


def test_remote_cleanup_checks_the_home_before_signalling(tmp_path):
    home = tmp_path / "h_test"
    remote = Remote({"host": "unused", "python": sys.executable, "work": str(tmp_path)},
                    "test", "unused.zarr", 12345, {}, tmp_path / "remote.log")
    args = shlex.split(remote.cleanup_args[-1])
    for matches, ignores_term in ((False, False), (True, False), (True, True)):
        actual_home = home if matches else tmp_path / "other"
        code = "import signal, time; " + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignores_term else "")
        code += "print('ready', flush=True); time.sleep(60)"
        proc = subprocess.Popen([sys.executable, "-c", code,
                                 "--home", str(actual_home)], stdout=subprocess.PIPE, text=True)
        try:
            assert proc.stdout.readline().strip() == "ready"
            (tmp_path / "h_test.pid").write_text(str(proc.pid))
            subprocess.run(args, check=True)
            if matches:
                assert proc.wait(timeout=5) == (-9 if ignores_term else -15)
            else:
                assert proc.poll() is None
        finally:
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=5)
