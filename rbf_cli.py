# rbf_cli.py
import sys
import logging

from rbf_args import parse_args
from rbf_logging import setup_logging
from rbf_runner import run_all


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    try:
        return run_all(args)
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
