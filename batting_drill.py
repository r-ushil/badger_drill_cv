from cover_drive_judge import CoverDriveJudge
import sys

def main(input_filename):
  judge = CoverDriveJudge(input_filename)
  judge.process_and_write_video()
  judge.cleanup_resources()


if __name__ == "__main__":
  main(sys.argv[1])

