from enum import Enum
from typing import Optional

class CatchingDrillError(Enum):
    BOUNCE_POINT_NOT_DETECTED = "not-detected/point-bounce"
    CATCH_POINT_NOT_DETECTED = "not-detected/point-catch"
    KATCHET_BOARD_NOT_DETECTED = "not-detected/katchet-board"
    POSE_NOT_DETECTED = "not-detected/pose"

    def get_err_code(self) -> str:
        return self.value

    def get_err_message(self) -> str:
        match self:
            case CatchingDrillError.BOUNCE_POINT_NOT_DETECTED:
                return "Failed to detect bounce point"
            case CatchingDrillError.CATCH_POINT_NOT_DETECTED:
                return "Failed to detect catch point"
            case CatchingDrillError.KATCHET_BOARD_NOT_DETECTED:
                return "Failed to detect Katchet board"
            case CatchingDrillError.POSE_NOT_DETECTED:
                return "Failed to detect pose"


class CatchingDrillResults():
    def __init__(
        self,
        speed: float = 0.0,
        max_height: float = 0.0,
        angle: float = 0.0,
        err: Optional[CatchingDrillError] = None
    ) -> None:
        self.speed = speed
        self.max_height = max_height
        self.angle = angle
        self.err = err
