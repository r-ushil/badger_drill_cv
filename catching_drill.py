from catching_judge import CatchingJudge
import sys

def main(input_filename):
	with CatchingJudge(input_filename) as judge:
		judge.process_and_write_video()

if __name__ == "__main__":
	main(sys.argv[1])