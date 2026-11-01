"""Bound native stderr/stdout on long training runs without blocking protocol pipes."""
import os
from pathlib import Path
import threading


class NativeLog:
    def __init__(self, path, limit=2*1024*1024):
        self.path,self.limit = Path(path),limit
        self.bounded = os.environ.get("RM_BOUNDED_NATIVE_LOGS") == "1"
        self.file = self.thread = self.writer = None
        self.error = None

    def __enter__(self):
        if not self.bounded:
            self.file = self.path.open("wb")
            self.output = self.file
            return self
        reader,self.writer = os.pipe()
        self.output = self.writer
        def drain():
            try:
                with os.fdopen(reader,"rb") as source:
                    output = self.path.open("wb")
                    size = 0
                    try:
                        while True:
                            chunk = source.read1(min(65536,self.limit))
                            if not chunk:break
                            if size+len(chunk) > self.limit:
                                output.close()
                                self.path.replace(self.path.with_suffix(self.path.suffix+".1"))
                                output = self.path.open("wb")
                                size = 0
                            output.write(chunk)
                            output.flush()
                            size += len(chunk)
                    finally:
                        output.close()
            except BaseException as error:
                self.error = error
        self.thread = threading.Thread(target=drain,daemon=True)
        self.thread.start()
        return self

    def spawned(self):
        if self.writer is not None:
            os.close(self.writer)
            self.writer = None

    def __exit__(self,*error):
        self.spawned()
        if self.file is not None:self.file.close()
        if self.thread is not None:
            self.thread.join(5)
            if self.thread.is_alive() and error[0] is None:
                raise RuntimeError("native log drain failed to finish")
        if self.error is not None and error[0] is None:
            raise RuntimeError("native log write failed") from self.error
