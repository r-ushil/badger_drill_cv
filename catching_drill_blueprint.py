from math import isnan
from flask import Blueprint, jsonify, request

from catching_judge import CatchingJudge
from point_projector import CameraIntrinsics

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

	cam = CameraIntrinsics(
		focal_len=4.3,
		sensor_w=4.2,
		sensor_h=5.6,
		image_h=1960.0,
		image_w=1080.0,
	)

	with CatchingJudge(obj_signed_url, cam, no_output=True) as judge:
		result = judge.process_and_write_video()

		if result.err is not None:
			err = result.get_error()
			return ','.join([str(int(0)), str(err), ""])

		advice1 = f"You caught the ball at {round(result.get_speed(), 2)}ms-1."
		advice2 = f"The ball bounced {round(result.get_max_height(), 2)}m high!"

		return ','.join([str(int(result.get_score())), advice1, advice2])
