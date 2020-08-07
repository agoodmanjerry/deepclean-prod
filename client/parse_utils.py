import argparse
import six
import configparser
import logging


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
		for key, val in config.items('config'):
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
						"only accepts one value".format(
							", ".join(val), key)
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


def build_parser(parent_parser):
	parser = argparse.ArgumentParser(
		parents=[parent_parser],
		conflict_handler='resolve'
	)
	parser.add_argument(
		'--file',
		type=str,
		action=ConfigParserAction,
		help='Path to config file'
	)
	return parser
