"""
Launch the Intelligent EDA dashboard.

    python app.py                              # start empty, upload in the UI
    python app.py --file data.xlsx             # preload a dataset
    python app.py --file data.xlsx --sheet S1  # pick an Excel sheet
"""

import argparse

from ui import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Intelligent EDA dashboard")
    parser.add_argument("--file", help="Dataset to open on startup (optional).")
    parser.add_argument("--sheet", help="Excel sheet name (default: the sheet with the most columns).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true", help="Enable Dash dev tools and hot reload.")
    args = parser.parse_args()

    app = create_app(args.file, args.sheet)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
