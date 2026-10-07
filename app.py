"""
Launch the Intelligent EDA dashboard.

    python app.py                              # start empty, upload in the UI
    python app.py --file data.xlsx             # preload a dataset
    python app.py --file data.xlsx --sheet S1  # pick an Excel sheet
"""

import argparse
import socket
import sys

from ui import create_app


def port_in_use(host: str, port: int) -> bool:
    """True if a server already answers on host:port (Windows lets a second one bind silently)."""

    target = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((target, port)) == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Intelligent EDA dashboard")
    parser.add_argument("--file", help="Dataset to open on startup (optional).")
    parser.add_argument("--sheet", help="Excel sheet name (default: the sheet with the most columns).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true", help="Enable Dash dev tools and hot reload.")
    args = parser.parse_args()

    if port_in_use(args.host, args.port):
        sys.exit(
            f"Port {args.port} is already in use: another dashboard is probably running there "
            f"(it answers at http://127.0.0.1:{args.port}/). Stop it, or start this one with --port {args.port + 1}."
        )

    app = create_app(args.file, args.sheet)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
