import parse_utils
import argparse

def get_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--a', required=True, type=int)
	parser.add_argument('--boolguy', action='store_true')
	parser.add_argument('--b', default=10)
	parser.add_argument('--c', default=6)
	return parser

parser = parse_utils.build_parser(get_parser())
args = ['--file', 'test_config.ini']
try:
	print('Before Error')
	parser.parse_args(args)
except:
	print('After Error')
	pass

parser = parse_utils.build_parser(get_parser())
args.extend(['--a', '12'])
flags = parser.parse_args(args)
print(flags)

parser = parse_utils.build_parser(get_parser())
args.extend(['--b', '-1'])
flags = parser.parse_args(args)
print(flags)