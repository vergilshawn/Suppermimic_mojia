"""Render policy/reference comparisons or serve the retained comparison website."""
import argparse
from pathlib import Path

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    r = sub.add_parser('render', help='Run deterministic CPU physics and render actual policy tracking.')
    r.add_argument('--checkpoint', required=True, type=Path)
    r.add_argument('--motion', required=True, nargs='+', help='Exact dataset clip names')
    r.add_argument('--out', required=True, type=Path)
    r.add_argument('--seconds', type=float, default=12)
    r.add_argument('--width', type=int, default=1920)
    r.add_argument('--height', type=int, default=1080)
    s = sub.add_parser('serve', help='Serve existing comparison files locally.')
    s.add_argument('--directory', type=Path, default=Path('web'))
    s.add_argument('--port', type=int, default=8000)
    a = p.parse_args(argv)
    if a.mode == 'serve':
        from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
        from functools import partial
        if not a.directory.is_dir(): p.error('Website directory does not exist')
        server = ThreadingHTTPServer(('127.0.0.1', a.port), partial(SimpleHTTPRequestHandler, directory=str(a.directory.resolve())))
        print(f'http://127.0.0.1:{a.port}', flush=True)
        try: server.serve_forever()
        except KeyboardInterrupt: pass
        finally: server.server_close()
        return
    if a.seconds <= 0 or min(a.width, a.height) < 64 or a.width % 2 or a.height % 2:
        p.error('Use positive duration and even image dimensions >=64')
    from .rendering import render
    render(a)
