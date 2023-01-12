from enum import Enum
from typing import Optional

class CatchingDrillErrorType(Enum):
    BOUNCE_POINT_NOT_DETECTED = "not-detected/point-bounce"
    CATCH_POINT_NOT_DETECTED = "not-detected/point-catch"
    KATCHET_BOARD_NOT_DETECTED = "not-detected/katchet-board"
    POSE_NOT_DETECTED = "not-detected/pose"

    def get_err_code(self) -> str:
        return self.value

    def get_err_message(self) -> str:
        match self:
            case CatchingDrillErrorType.BOUNCE_POINT_NOT_DETECTED:
                return "Failed to detect bounce point"
            case CatchingDrillErrorType.CATCH_POINT_NOT_DETECTED:
                return "Failed to detect catch point"
            case CatchingDrillErrorType.KATCHET_BOARD_NOT_DETECTED:
                return "Failed to detect Katchet board"
            case CatchingDrillErrorType.POSE_NOT_DETECTED:
                return "Failed to detect pose"

class CatchingDrillError(Exception):
    def __init__(self, err_type: CatchingDrillErrorType) -> None:
        super().__init__(err_type.get_err_message())

class CatchingDrillResults():
    def __init__(
        self,
        speed: float = 0.0,
        max_height: float = 0.0,
        err: Optional[CatchingDrillError] = None
    ) -> None:
        self.speed = speed
        self.max_height = max_height
        self.err = err
    
    def get_score(self):
        max_speed = 29
        min_speed = 10
        speed_range = max_speed - min_speed

        max_height = 2.8
        min_heigth = 0.2
        height_range = max_height - min_heigth

        scaled_speed = self.speed / speed_range
        scaled_heigth = self.max_height / height_range

        print("Speed score:", scaled_speed)
        print("Height score:", scaled_heigth)
        
        height_contribution = 0.2
        speed_contibution = 0.8

        score = int((height_contribution * scaled_heigth + speed_contibution * scaled_speed) * 100)

        print("Final score:", score)

        return score

    def get_max_height(self):
        return self.max_height
    
    def get_speed(self):
        return self.speed

    def get_error(self):
        return self.err
