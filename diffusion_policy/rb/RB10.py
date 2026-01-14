import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class RB10(DHRobot):
    """
    Class that models a Rainbow Robotics RB10 manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``RB10()`` is an object which models a Unimation Puma560 robot and
    describes its kinematic and dynamic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.RB10()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration
    - qr, arm horizontal along x-axis

    .. note::
        - SI units are used.

    :References:

        - `Parameters for calculations of kinematics and dynamics <https://www.universal-robots.com/articles/ur/parameters-for-calculations-of-kinematics-and-dynamics>`_

    :sealso: :func:`UR4`, :func:`UR10`


    .. codeauthor:: Peter Corke
    
    .. constructor:: Seunghwan Um
    """  # noqa

    def __init__(self, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym

            zero = sym.zero()
            pi = sym.pi()
        else:
            from math import pi

            zero = 0.0

        deg = pi / 180
        inch = 0.0254

        # robot length values (metres)
        # a = [0, -0.42500, -0.39225, 0, 0, 0]
        # d = [0.1692, 0, 0, 0.1107, 0.1107, 0.0967]
        # alpha = [pi/2, zero, zero, pi / 2, -pi / 2, zero]
        # offset = [0,-pi/2,0,-pi/2,0,0]

        a = [zero,     0.6127, 0.57015, zero, zero, zero]
        d = [0.1970, -0.1875, 0.1484, -0.11715, 0.11715, -0.1153]
        # d = [0.1970, -0.0391, 0, -0.11715, 0.11715, -0.1153]
        # d = [0.1970, 0, -0.0, -0.15625, 0.11715, -0.1153]

        alpha = [-pi/2,  zero,       zero,   pi/2,  -pi/2,   pi/2]
        offset= [zero,     -pi/2,     zero,   pi/2,      zero,      zero]
        
        # mass data, no inertia available
        mass = [8.385, 15.648, 5.717, 2.062, 2.061, 0.406]
        center_of_mass = [
            [-0.000029, -0.007927, -0.042724],
            [ 0.000013, -0.155431,  0.225425],
            [-0.000048, -0.013820,  0.325855],
            [ 0.000094, -0.113946,  0.026460],
            [-0.000094, -0.026433, -0.003237],
            [-0.000196, -0.092539, -0.000271],
        ]

        # 모든 관성 항목은 단위 [kg·m²]로 변환 (단위변환: mm² → m², 1e-6 곱함)
        inertia = [
            [0.027564395, 0.027896623, 0.026229737, 0.000028342, 0.003057757, 0.000000330],
            [1.143769668, 1.137336583, 0.051104148, 0.000037669, 0.025345902, 0.000020190],
            [0.312935818, 0.312185527, 0.011429150, 0.000011605, 0.005264922, 0.000009306],
            [0.002710874, 0.002715201, 0.002157325, 0.000011241, 0.000144363, -0.000008028],
            [0.002706211, 0.002153873, 0.002712066, -0.000008022, 0.000141647, 0.000011238],
            [0.000344938, 0.000532281, 0.000339793, 0.000000658, -0.000000060, 0.000000580],
        ]
        links = []

        for j in range(6):
            link = RevoluteDH(
                d=d[j], a=a[j], alpha=alpha[j], offset=offset[j], m=mass[j], r=center_of_mass[j], G=1, I=inertia[j]
            )
            links.append(link)

        super().__init__(
            links,
            name="RB10",
            manufacturer="Rainbow Robotics",
            keywords=("dynamics", "symbolic"),
            symbolic=symbolic,
        )

        self.qr = np.array([180, 0, 0, 0, 90, 0]) * deg
        self.qz = np.zeros(6)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)



if __name__ == "__main__":  # pragma nocover

    RB10 = RB10(symbolic=False)
    print(RB10)
    print('▗▄▄▄▖▗▄▄▄▖▗▖  ▗▖▗▄▄▖  ▗▄▖ ▗▄▄▖  ▗▄▖ ▗▄▄▖▗▖  ▗▖     ▗▄▄▖▗▄▄▄▖▗▖  ▗▖▗▄▄▄▖▗▄▄▖  ▗▄▖▗▄▄▄▖▗▄▄▄▖▗▄▄▄ ')
    print('  █  ▐▌   ▐▛▚▞▜▌▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▝▚▞▘     ▐▌   ▐▌   ▐▛▚▖▐▌▐▌   ▐▌ ▐▌▐▌ ▐▌ █  ▐▌   ▐▌  █')
    print('  █  ▐▛▀▀▘▐▌  ▐▌▐▛▀▘ ▐▌ ▐▌▐▛▀▚▖▐▛▀▜▌▐▛▀▚▖ ▐▌      ▐▌▝▜▌▐▛▀▀▘▐▌ ▝▜▌▐▛▀▀▘▐▛▀▚▖▐▛▀▜▌ █  ▐▛▀▀▘▐▌  █')
    print('  █  ▐▙▄▄▖▐▌  ▐▌▐▌   ▝▚▄▞▘▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌ ▐▌      ▝▚▄▞▘▐▙▄▄▖▐▌  ▐▌▐▙▄▄▖▐▌ ▐▌▐▌ ▐▌ █  ▐▙▄▄▖▐▙▄▄▀')
    # print(RB10.dyntable())