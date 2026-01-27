#!/usr/bin/env python3
"""RTDE proxy server for on-demand start/stop from another process."""
import argparse
import os
import signal
import sys
import time
from multiprocessing.managers import BaseManager, SharedMemoryManager

from diffusion_policy.real_world.rtde_interpolation_controller import RTDEInterpolationController


class RTDEProxy:
    def __init__(
        self,
        robot_ip,
        frequency,
        lookahead_time,
        gain,
        max_pos_speed,
        max_rot_speed,
        launch_timeout,
        tcp_offset_pose,
        get_max_k,
    ):
        self._robot_ip = robot_ip
        self._frequency = frequency
        self._lookahead_time = lookahead_time
        self._gain = gain
        self._max_pos_speed = max_pos_speed
        self._max_rot_speed = max_rot_speed
        self._launch_timeout = launch_timeout
        self._tcp_offset_pose = tcp_offset_pose
        self._get_max_k = get_max_k
        self._ctrl = None

    def start_rtde(self):
        if self._ctrl is not None:
            return True
        shm_manager = SharedMemoryManager()
        shm_manager.start()
        self._ctrl = RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip=self._robot_ip,
            frequency=self._frequency,
            lookahead_time=self._lookahead_time,
            gain=self._gain,
            max_pos_speed=self._max_pos_speed,
            max_rot_speed=self._max_rot_speed,
            launch_timeout=self._launch_timeout,
            tcp_offset_pose=self._tcp_offset_pose,
            payload_mass=None,
            payload_cog=None,
            joints_init=None,
            joints_init_speed=1.05,
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=self._get_max_k,
        )
        self._shm_manager = shm_manager
        self._ctrl.start(wait=False)
        t0 = time.time()
        while True:
            if self._ctrl.is_ready:
                return True
            if not self._ctrl.is_alive():
                return False
            if time.time() - t0 > self._launch_timeout:
                return False
            time.sleep(0.1)

    def stop_rtde(self):
        if self._ctrl is None:
            return True
        try:
            self._ctrl.stop(wait=True)
        finally:
            self._ctrl = None
            if hasattr(self, "_shm_manager") and self._shm_manager is not None:
                try:
                    self._shm_manager.shutdown()
                except Exception:
                    pass
                self._shm_manager = None
        return True

    def is_ready(self):
        return self._ctrl is not None and self._ctrl.is_ready

    def is_alive(self):
        return self._ctrl is not None and self._ctrl.is_alive()

    def get_state(self, k=1):
        if self._ctrl is None:
            return None
        return self._ctrl.get_state(k=k)

    def schedule_waypoint(self, pose, target_time):
        if self._ctrl is None:
            return False
        self._ctrl.schedule_waypoint(pose=pose, target_time=target_time)
        return True

    def ping(self):
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_ip", required=True)
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--authkey", default="rb10_rtde")
    parser.add_argument("--frequency", type=float, default=125.0)
    parser.add_argument("--lookahead_time", type=float, default=0.1)
    parser.add_argument("--gain", type=float, default=300.0)
    parser.add_argument("--max_pos_speed", type=float, default=0.25)
    parser.add_argument("--max_rot_speed", type=float, default=0.6)
    parser.add_argument("--launch_timeout", type=float, default=5.0)
    parser.add_argument("--tcp_offset_pose", default="0,0,0.13,0,0,0")
    parser.add_argument("--get_max_k", type=int, default=30)
    args = parser.parse_args()

    tcp_offset_pose = [float(x) for x in args.tcp_offset_pose.split(",")]

    class _Manager(BaseManager):
        pass

    def _make_proxy():
        return RTDEProxy(
            robot_ip=args.robot_ip,
            frequency=args.frequency,
            lookahead_time=args.lookahead_time,
            gain=args.gain,
            max_pos_speed=args.max_pos_speed,
            max_rot_speed=args.max_rot_speed,
            launch_timeout=args.launch_timeout,
            tcp_offset_pose=tcp_offset_pose,
            get_max_k=args.get_max_k,
        )

    _Manager.register("RTDEProxy", callable=_make_proxy)

    mgr = _Manager(address=("127.0.0.1", args.port), authkey=args.authkey.encode("utf-8"))
    server = mgr.get_server()

    def _shutdown(signum, frame):
        os._exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    server.serve_forever()


if __name__ == "__main__":
    main()
