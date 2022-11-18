from math import isnan
from flask import Blueprint, jsonify, request

from cover_drive_judge import CoverDriveJudge
from storage import get_object_signed_url

cover_drive_blueprint = Blueprint('cover_drive_drill', __name__)

@cover_drive_blueprint.route("/cover-drive-drill", methods=["GET"])
def process_batting_drill_video():
	obj_name = request.args.get('video_object_name', None)

	if obj_name == None:
		for key, value in request.args.items():
			print(key, value)
		return 'Missing video_object_name', 400

	obj_signed_url = get_object_signed_url(obj_name)

	with CoverDriveJudge(obj_signed_url, no_output=True) as judge:
		(score, advice1, advice2) = judge.process_and_write_video()

		if isnan(score):
			score = 0

		return jsonify(score=score, advice=[advice1, advice2])
