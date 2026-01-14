import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class RB5(DHRobot):
    """
    Class that models a Rainbow Robotics RB5 manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``RB5()`` is an object which models a Unimation Puma560 robot and
    describes its kinematic and dynamic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.RB5()
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
        
        
        a =     [0,   0.42500,    0.39225,      0,      0,      0]
        d =     [0.1692,    0,          0,-0.1107, 0.1107,-0.0967-0.003]
        alpha = [-pi/2,  zero,       zero,   pi/2,  -pi/2,   pi/2]
        offset= [0,     -pi/2,          0,   pi/2,      0,      0]
        # mass data, no inertia available
        mass = [5.588336, 9.503399, 3.202978, 1.356780, 1.356780, 0.171541]
        center_of_mass = [
            [0.000012, -0.002613, 0.060352],
            [0, -0.114981, 0.211114],
            [0.000004, -0.018194, 0.219836],
            [-0.000024, -0.106297, 0.029356],
            [0.000024, -0.029356, -0.044022],
            [0.001715, -0.080700, -0.000180],
        ]
        inertia = [
            [0.041227473290685, 0.040817666882974, 0.012209015209776,   -0.000007297191574, -0.001034629955434, -0.000008403038448],
            [0.949319006774135, 0.819596599298864, 0.146822420639023,   -0.000000000004452, -0.234095204410459,  0.000009076795146],
            [0.245139908783287, 0.243950697411086, 0.006921912119249,   -0.000000596636031, -0.015628137286249,  0.000013885046111],
            [0.017845581259381, 0.002477039039656, 0.016422745831595,    0.000002891849882, -0.004375863929516, -0.000000525951420],
            [0.002541474364616, 0.001118638936830, 0.002477039039657,   -0.000000525951420,  0.000033215608299, -0.000000669457895],
            [0.001442738077985, 0.000213622367571, 0.001446789772376,   -0.000002617454254,  0.000002748095325, -0.000000228220422]        
        ]
        links = []

        for j in range(6):
            link = RevoluteDH(
                d=d[j], a=a[j], alpha=alpha[j], offset=offset[j], m=mass[j], r=center_of_mass[j], G=1, I=inertia[j]
            )
            links.append(link)

        super().__init__(
            links,
            name="RB5",
            manufacturer="Rainbow Robotics",
            keywords=("dynamics", "symbolic"),
            symbolic=symbolic,
        )

        self.qr = np.array([180, 0, 0, 0, 90, 0]) * deg
        self.qz = np.zeros(6)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    rb5 = RB5(symbolic=False)
    print(rb5)