"""Camera reconstruction and display-time boundaries, independent of model inference."""
import unittest
from types import SimpleNamespace
import numpy as np
from tools.training.render_mcap import ALIGN, bracket, reconstruct_camera, rotation, quaternion


class RenderCameraTests(unittest.TestCase):
    def test_recovers_absolute_capture_pose_from_multiple_world_planes(self):
        calibration=SimpleNamespace(K=[800.,0,640.,0,800.,360.,0,0,1.])
        origin=np.array([1.2,-.7,.4]);r=rotation([.04,-.12,.06,.99])
        plates=[]
        for z,tilt in [(4.,0.),(4.3,.2)]:
            plane=np.array([[-.12,-.05,z],[.12,-.05,z+tilt],[.12,.05,z+tilt],[-.12,.05,z]])
            world=plane@r.T+origin
            pixels=plane[:,:2]/plane[:,2,None]*800+[640,360]
            plates.append(dict(corners_world=world.tolist(),ideal_corners_px=pixels.tolist()))
        camera,error=reconstruct_camera(plates,calibration)
        np.testing.assert_allclose(camera['translation'],ALIGN.T@origin,atol=1e-5)
        np.testing.assert_allclose(rotation(camera['rotation_xyzw']),ALIGN.T@r@np.diag([1,-1,-1]),atol=1e-5)
        self.assertLess(error,1e-4)

    def test_missing_truth_rays_are_not_filled_with_estimates(self):
        with self.assertRaisesRegex(ValueError,'no capture-time truth'):
            reconstruct_camera([],SimpleNamespace(K=[800,0,640,0,800,360,0,0,1]))

    def test_brackets_do_not_extrapolate_or_bridge_missing_control_steps(self):
        self.assertEqual(bracket({0:'a',10_000_000:'b'},5_000_000),('a','b',.5))
        with self.assertRaisesRegex(ValueError,'no physical bracket'):
            bracket({10:'a',20:'b'},9)
        with self.assertRaisesRegex(ValueError,'exceeds 10 ms'):
            bracket({0:'a',20_000_000:'b'},5_000_000)

    def test_matrix_quaternion_roundtrip_at_half_turn(self):
        r=rotation([1,0,0,0]);np.testing.assert_allclose(rotation(quaternion(r)),r,atol=1e-12)


if __name__=='__main__':unittest.main()
