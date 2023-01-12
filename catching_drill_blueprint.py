from math import isnan
from flask import Blueprint, jsonify, request

from catching_judge import CatchingJudge

from storage import get_object_signed_url

catching_drill_blueprint = Blueprint('catching_drill', __name__)

@catching_drill_blueprint.route("/catching-drill", methods=["POST"])
def process_catching_drill_video():
	req_json = request.json
	obj_name = req_json.get('video_object_name', None)

	if type(obj_name) is not str:
		for key, value in request.args.items():
			print(key, value)
		return 'Missing video_object_name', 400

	obj_signed_url = get_object_signed_url(obj_name)

	with CatchingJudge(obj_signed_url, no_output=True) as judge:
		result = judge.process_and_write_video()

		if result.err is not None:
			return jsonify(
				error_code=result.err.get_err_code(),
				error_message=result.err.get_err_message(),
			), 400

		return jsonify(
            score=result.get_score(),
        )
