"""HCN -> RINEX converter (CHCNAV). Drag & drop or browse. Internal use."""
from __future__ import annotations

import ctypes
import io
import os
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

APP_DIR = Path(__file__).resolve().parent
BIN = APP_DIR / "bin"
LOG_FILE = APP_DIR / "last_conversion.log"


def write_conversion_log(
    hcn: Path,
    *,
    ok: bool,
    msg: str,
    dll_output: str = "",
) -> None:
    lines = [
        f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"input: {hcn.resolve()}",
        f"output_dir: {hcn.parent.resolve()}",
        f"status: {'OK' if ok else 'FAILED'}",
        "message:",
        msg or "(none)",
    ]
    if dll_output.strip():
        lines.extend(["", "dll_output:", dll_output.rstrip()])
    text = "\n".join(lines) + "\n"
    LOG_FILE.write_text(text, encoding="utf-8")


def open_folder(path: Path) -> None:
    folder = path.resolve()
    if not folder.is_dir():
        folder = folder.parent
    if sys.platform == "win32":
        os.startfile(folder)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        os.system(f'open "{folder}"')
    else:
        os.system(f'xdg-open "{folder}"')


def convert_hcn(hcn: Path) -> tuple[bool, str, str]:
    if hcn.suffix.lower() != ".hcn":
        return False, "Only .hcn files are supported", ""
    if not hcn.is_file():
        return False, f"File not found: {hcn}", ""
    for name in ("GNSSQualityCheck.dll", "RinexOption.xml"):
        if not (BIN / name).exists():
            return False, f"Missing {name} in {BIN}", ""

    out_dir = hcn.parent
    before = {p.resolve() for p in out_dir.glob("*") if p.is_file()}
    cwd = os.getcwd()
    dll_output = ""
    ok = False
    try:
        os.chdir(BIN)
        os.add_dll_directory(str(BIN))
        dll = ctypes.CDLL(str(BIN / "GNSSQualityCheck.dll"))
        fn = dll["?SolveHcnToRinexEx@GNSSQualityCheck@@SA_NPEBD00NNH_N@Z"]
        fn.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_bool,
        ]
        fn.restype = ctypes.c_bool
        opt = str(BIN / "RinexOption.xml").encode()
        capture = io.StringIO()
        with open(os.devnull, "w") as devnull:
            old_out, old_err = sys.stdout, sys.stderr

            class _Tee:
                def __init__(self, stream, sink: io.StringIO) -> None:
                    self._stream = stream
                    self._sink = sink

                def write(self, data: str) -> int:
                    if data:
                        self._sink.write(data)
                    try:
                        return self._stream.write(data)
                    except OSError:
                        return len(data)

                def flush(self) -> None:
                    self._sink.flush()
                    try:
                        self._stream.flush()
                    except OSError:
                        pass

            sys.stdout = _Tee(devnull, capture)
            sys.stderr = _Tee(devnull, capture)
            try:
                ok = bool(
                    fn(
                        str(hcn).encode(),
                        str(out_dir).encode(),
                        opt,
                        0.0,
                        0.0,
                        0,
                        False,
                    )
                )
            finally:
                sys.stdout, sys.stderr = old_out, old_err
        dll_output = capture.getvalue()
    except OSError as e:
        return False, str(e), dll_output
    finally:
        os.chdir(cwd)

    if not ok:
        return False, "Conversion failed (DLL returned false)", dll_output

    new_files = [
        p for p in out_dir.glob("*") if p.is_file() and p.resolve() not in before
    ]
    rnx = [p for p in new_files if p.suffix.lower() in {".26o", ".26n", ".rnx"}]
    if not rnx:
        rnx = list(out_dir.glob(f"{hcn.stem}.*"))
    msg = "\n".join(p.name for p in sorted(rnx)[:12]) if rnx else str(out_dir)
    return True, msg, dll_output


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("HCN → RINEX")
        self.geometry("480x260")
        self.resizable(False, False)
        self._current_path: Path | None = None
        self._last_output_dir: Path | None = None

        self.drop = tk.Label(
            self,
            text="Drag & drop a .hcn file here\nor click to browse",
            relief="groove",
            bg="#f0f4f8",
            font=("Segoe UI", 11),
            height=6,
            cursor="hand2",
        )
        self.drop.pack(fill="both", expand=True, padx=16, pady=(16, 8))
        self.drop.bind("<Button-1>", lambda _e: self._browse())

        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, wraplength=440).pack(padx=16, pady=(0, 8))

        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", padx=16, pady=(0, 16))
        self.open_btn = ttk.Button(
            btn_row,
            text="Open output folder",
            command=self._open_output_folder,
            state="disabled",
        )
        self.open_btn.pack(side="left")
        ttk.Button(btn_row, text="Exit", command=self.destroy).pack(side="right")

        try:
            import windnd

            windnd.hook_dropfiles(self, func=self._on_drop)
        except ImportError:
            self.status.set("Ready (install windnd for drag & drop: pip install windnd)")

    def _on_drop(self, files: list[bytes]) -> None:
        if not files:
            return
        path = Path(files[0].decode("mbcs" if sys.platform == "win32" else "utf-8"))
        self._run(path)

    def _browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select HCN file",
            filetypes=[("CHCNAV HCN", "*.hcn"), ("All files", "*.*")],
        )
        if path:
            self._run(Path(path))

    def _open_output_folder(self) -> None:
        if self._last_output_dir is not None:
            open_folder(self._last_output_dir)

    def _run(self, path: Path) -> None:
        self._current_path = path
        self.status.set(f"Converting: {path.name}...")
        self.drop.config(state="disabled")
        self.open_btn.config(state="disabled")

        def work() -> None:
            try:
                ok, msg, dll_output = convert_hcn(path)
                write_conversion_log(path, ok=ok, msg=msg, dll_output=dll_output)
                self.after(0, lambda: self._done(ok, msg))
            except Exception as exc:
                write_conversion_log(
                    path,
                    ok=False,
                    msg=f"Exception: {exc}",
                    dll_output="",
                )
                self.after(0, lambda: self._done(False, str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _done(self, ok: bool, msg: str) -> None:
        self.drop.config(state="normal")
        path = self._current_path
        out_dir = path.parent if path is not None else APP_DIR
        if ok:
            self._last_output_dir = out_dir
            self.open_btn.config(state="normal")
            self.status.set(f"Done — {out_dir}")
            messagebox.showinfo(
                "Conversion complete",
                f"RINEX written to:\n{out_dir}\n\n{msg}\n\nLog: {LOG_FILE}",
            )
        else:
            self.status.set(f"Error — {msg}")
            messagebox.showerror("Conversion failed", f"{msg}\n\nLog: {LOG_FILE}")


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
