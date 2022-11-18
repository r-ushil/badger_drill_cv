import os
import math

from flask import Flask, request
from cover_drive_judge import CoverDriveJudge
from urllib.parse import unquote

app = Flask(__name__)

def processVideo(url):
	clean_url = unquote(url)
	dodge_fix = clean_url[:77] + "%2F" + clean_url[78:]

	with CoverDriveJudge(dodge_fix) as judge:
		(averageScore, advice1, advice2) = judge.process_and_write_video()
		if math.isnan(averageScore):
			averageScore = 0
		return (averageScore, advice1, advice2)

@app.route("/", methods=["GET"])
def generate_response():
	url = request.args.get('url', None)
	(score, comment1, comment2) = processVideo(url)
	return ','.join([str(int(score * 100)), comment1, comment2])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
