from math import isnan
from flask import Blueprint, jsonify, request

from catching_judge import CatchingJudge

from storage import get_object_signed_url

catching_drill_blueprint = Blueprint('catching_drill', __name__)

@catching_drill_blueprint.route("/catching-drill", methods=["GET"])
def process_catching_drill_video():
	obj_name = request.args.get('video_object_name', None)

	if obj_name == None:
		for key, value in request.args.items():
			print(key, value)
		return 'Missing video_object_name', 400

	obj_signed_url = get_object_signed_url(obj_name)

	with CatchingJudge(obj_signed_url, no_output=True) as judge:
		result = judge.process_and_write_video()

		if result.err is not None:
			return ','.join([str(int(0)), "", ""])

		return ','.join([str(int(result.get_score())), "", ""])
