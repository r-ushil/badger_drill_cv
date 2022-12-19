from numpy import array, average, float64, ndarray

KATCHET_BOX_TOP_L = [.0, -.05, .45]
KATCHET_BOX_TOP_R = [.58, -.05, .45]
KATCHET_BOX_BOT_L = [.0, .0, .0]
KATCHET_BOX_BOT_R = [.58, .0, .0]

class KatchetBoard():
    __vertices_2d: ndarray[(4, 2), float64]
    __vertices_3d: ndarray[(4, 3), float64]

    def __init__(self, vertices_2d: ndarray[(4, 2), float64], vertices_3d: ndarray[(4, 3), float64]):
        self.__vertices_2d = vertices_2d
        self.__vertices_3d = vertices_3d

    @classmethod
    def from_vertices_2d(board, vertices_2d: ndarray[(4, 2), float64]) -> "KatchetBoard":
        center_2d = average(vertices_2d, axis=0)
        deltas_2d = vertices_2d - center_2d

        vertices_3d = array([board.__pick_vertex_3d_from_delta(delta_2d) for delta_2d in deltas_2d])

        return KatchetBoard(vertices_2d, vertices_3d)

    @staticmethod
    def __pick_vertex_3d_from_delta(vertex_2d: ndarray[(2,), float64]):
        [x, y] = vertex_2d

        if y < 0:
            return KATCHET_BOX_TOP_L if x < 0 else KATCHET_BOX_TOP_R
        else:
            return KATCHET_BOX_BOT_L if x < 0 else KATCHET_BOX_BOT_R

    def get_vertices_2d(self) -> ndarray[(4, 2), float64]:
        return self.__vertices_2d.astype('float64')

    def get_vertices_3d(self) -> ndarray[(4, 3), float64]:
        return self.__vertices_3d.astype('float64')
