import argparse
import os
import configparser
import logging

from deepclean_prod import config


logger = logging.getLogger(__name__)


_BOOL_ACTIONS = (argparse._StoreTrueAction, argparse._StoreFalseAction)


class ConfigParserAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        actions = parser._actions[1:]

        def find_action(name):
            return [action for action in actions if action.dest == name][0]

        allowed_args = [action.dest for action in actions]
        required_args = [action for action in actions if action.required]

        config = configparser.ConfigParser()
        config.BOOLEAN_STATES = {b: True for b in ["yes", "true", "t", "1"]}
        config.read(values)

        args = []
        for key, val in config.items("config"):
            if key not in allowed_args:
                logger.warning(f"Warning: Do not recognize key {key}")
            if key in [action.dest for action in required_args]:
                # remove requirement from arg so parser doesn't
                # complain that it didn't show up at the command
                # line
                action.required = False
                required_args.remove(key)

            action = find_action(key)
            if isinstance(action, _BOOL_ACTIONS):
                try:
                    boolval = config["config"].getboolean(key)
                except ValueError as e:
                    # TODO: add message using reraise
                    message = (
                        "Argument {} expects a boolean value, found "
                        "value {}".format(key, val)
                    )
                    raise e

                if action.const == boolval:
                    # only need to add the flag if the action created
                    # by the flag and its config boolean match,
                    # otherwise the appropriate default kicks in
                    args.append(f"--{key}")
            else:
                if not val:
                    continue

                args.append(f"--{key}")
                val = val.split(", ")
                if len(val) > 1 and action.nargs != "+":
                    raise ValueError(
                        "Provided multiple values {} to arg {} which "
                        "only accepts one value".format(", ".join(val), key)
                    )
                args.extend(val)

        # temporarily dodge required args in case they
        # get provided at the command line
        for action in required_args:
            action.required = False
        parser.parse_args(args, namespace=namespace)
        for action in required_args:
            action.required = True

        setattr(namespace, self.dest, values)


class ChannelListAction(argparse.Action):
    '''
    Allows user to either specify explicit space separated
    channel names, or a single file containing \n separated
    channel names
    '''
    def __call__(self, parser, namespace, values, option_string=None):
        # TODO: put in init, just to lazy to look up the
        # init signature at the moment
        assert self.nargs == "+"

        if len(values) == 1 and os.path.exists(values[0]):
            with open(values[0], 'r') as f:
                values = f.read().split("\n")
        setattr(namespace, self.dest, values)


def build_parser(parent_parser):
    '''
    quick utility function for tacking a config option
    on to any existing parser
    '''
    parser = argparse.ArgumentParser(
        parents=[parent_parser], conflict_handler="resolve"
    )
    parser.add_argument(
        "--file", type=str, action=ConfigParserAction, help="Path to config file"
    )
    return parser


def get_client_parser():
    parser = argparse.ArgumentParser()

    # Dataset arguments
    # TODO: can we expose channel-wise inputs on the server side and
    # do concatenation/windowing there as custom backend? Would prevent
    # unnecessary data transfer. Possibly even do post processing there
    # and just send back one sample that's at the center of the batch?
    # Will depend on extent to which network time is bottlenecking
    parser.add_argument(
        "--clean-t0",
        help="GPS of the first sample",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--clean-duration",
        help="Duration of frame",
        type=int,
        required=True
    )
    parser.add_argument(
        "--chanslist",
        help="Path to channel list",
        nargs="+",
        action=ChannelListAction,
        type=str
    )
    parser.add_argument(
        "--fs",
        help="Sampling frequency",
        default=config.DEFAULT_SAMPLE_RATE,
        type=float,
    )

    # Timeseries arguments
    # TODO: can we save these as metadata to the model
    # and read them as a ModelMetadataRequest? Kernel
    # size we can even infer from input shape and fs
    parser.add_argument(
        "--clean-kernel",
        help="Length of each segment in seconds",
        default=config.DEFAULT_CLEAN_KERNEL,
        type=float,
    )
    parser.add_argument(
        "--clean-stride",
        help="Stride between segments in seconds",
        default=config.DEFAULT_CLEAN_STRIDE,
        type=float,
    )
    parser.add_argument(
        "--pad-mode",
        help="Padding mode",
        default=config.DEFAULT_PAD_MODE,
        type=str
    )

    # Post-processing arguments
    parser.add_argument(
        "--window",
        help="Window to apply to overlap-add",
        default=config.DEFAULT_WINDOW,
        type=str,
    )

    # Input/output arguments
    parser.add_argument("--ppr-file", help="Path to preprocessing setting", type=str)
    parser.add_argument("--out-channel", help="Name of output channel", type=str)
    parser.add_argument("--log", help="Log file", type=str)

    # Server arguments
    parser.add_argument(
        "--url",
        help="Server url",
        default="localhost:8001",
        type=str
    )
    parser.add_argument(
        "--model-name",
        help="Name of inference model",
        required=True,
        type=str
    )
    parser.add_argument(
        "--model-version",
        help="Model version to use",
        default=1,
        type=int
    )
    # TODO: can we just get this from the input_shape until we
    # get dynamic batching working?
    parser.add_argument(
        "--batch-size",
        help="Number of windows to infer on at once",
        required=True,
        type=int,
    )
    return build_parser(parser)
