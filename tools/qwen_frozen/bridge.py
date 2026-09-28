"""Python 3.8 DT / Python 3.10 frozen Qwen bridge. No trainable state crosses it."""
import atexit
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time

import numpy as np
import torch
from protocol import send, receive


class FrozenQwenClient:
    def __init__(self, source, device, worker_python, timeout=600,
                 worker_script='worker.py', **settings):
        if device.type != 'cuda':
            raise ValueError('Qwen feature extraction requires a CUDA device')
        self.device = device
        self.directory = Path(tempfile.mkdtemp(prefix='dt-qwen-%s-' % os.getpid(), dir='/dev/shm'))
        self.sock, peer = socket.socketpair()
        self.sock.settimeout(timeout)
        self.process = None
        self.arrays = {}
        self.log = (self.directory/'worker.log').open('w')
        try:
            self.process = subprocess.Popen([worker_python, str(source/worker_script),
                '--fd', str(peer.fileno()), '--parent', str(os.getpid()),
                '--device', str(device), '--directory', str(self.directory)],
                pass_fds=(peer.fileno(),), stdin=subprocess.DEVNULL,
                stdout=self.log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                         TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1'))
            peer.close()
            send(self.sock, settings)
            self.receipt = self._response()
            print('Frozen Qwen worker ready: %s' % self.receipt, flush=True)
        except BaseException:
            peer.close()
            self.close()
            raise
        atexit.register(self.close)

    def _response(self):
        try:
            response = receive(self.sock)
        except (OSError, RuntimeError) as exc:
            log = self.directory/'worker.log'
            detail = log.read_text(errors='replace')[-12000:] if log.exists() else ''
            raise RuntimeError('Frozen Qwen connection failed: %s\n%s' % (exc, detail)) from exc
        if 'error' in response:
            raise RuntimeError('Frozen Qwen worker: '+response['error'])
        return response

    def _allocate(self, name, shape):
        current = self.arrays.get(name)
        if current is None or current.shape != shape:
            self.arrays[name] = np.lib.format.open_memmap(
                str(self.directory/(name+'.npy')), mode='w+', dtype=np.float32, shape=shape)
        return self.arrays[name]

    def __call__(self, images):
        if images.device != self.device or images.ndim != 4 or len(images) % 6:
            raise ValueError('Worker device or six-camera batch mismatch')
        n, c, h, w = images.shape
        if c != 3 or h % 32 or w % 32:
            raise ValueError('Qwen native patch/merge requires height and width divisible by 32')
        start = time.monotonic()
        image = self._allocate('input', tuple(images.shape))
        image[:] = images.detach().float().cpu().numpy()
        visual = self._allocate('visual', (n, 1024, h//32, w//32))
        language = self._allocate('language', (n//6, 1, 1024))
        send(self.sock, dict(command='extract', shape=list(images.shape)))
        self.last_stats = self._response()
        # Copies complete before the worker may overwrite its next result.
        result = (torch.from_numpy(visual).to(device=self.device, copy=True),
                  torch.from_numpy(language).to(device=self.device, copy=True))
        self.last_stats['bridge_seconds'] = time.monotonic() - start
        return result

    def close(self):
        process = getattr(self, 'process', None)
        if process is not None and process.poll() is None:
            try:
                send(self.sock, dict(command='close'))
                process.wait(timeout=5)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
        if hasattr(self, 'sock'):
            self.sock.close()
        if hasattr(self, 'log') and not self.log.closed:
            self.log.close()
        self.arrays = {}
        if hasattr(self, 'directory'):
            shutil.rmtree(str(self.directory), ignore_errors=True)
