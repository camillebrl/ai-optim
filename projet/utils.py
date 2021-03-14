import argparse
import constants as CN
import logging
import sys


def init_logging():
    logging_handlers = []
    logging_handlers.append(logging.StreamHandler(sys.stdout))
    logging_handlers.append(logging.FileHandler(CN.LOG_FILENAME))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=logging_handlers, )


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10")
    return parser
