from math import isnan
from flask import Blueprint, request

from cover_drive_judge import CoverDriveJudge
from urllib.parse import unquote

cover_drive_blueprint = Blueprint('batting_drill', __name__)

@cover_drive_blueprint.route("/", methods=["POST"])
def process_batting_drill_video():
	url = request.args.get('url', None)
	(score, comment1, comment2) = processVideo(url)
	return ','.join([str(int(score * 100)), comment1, comment2])

def processVideo(url):
    clean_url = unquote(url)
    dodge_fix = clean_url[:77] + "%2F" + clean_url[78:]

    with CoverDriveJudge(dodge_fix) as judge:
        (averageScore, advice1, advice2) = judge.process_and_write_video()
        if isnan(averageScore):
            averageScore = 0
        return (averageScore, advice1, advice2)