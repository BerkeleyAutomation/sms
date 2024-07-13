from yumiplanning.yumi_kinematics import YuMiKinematics as YK
from yumiplanning.yumi_planner import Planner
from yumirws.yumi import YuMi
from autolab_core import RigidTransform, Box, Point
import numpy as np
import dexnerf.capture.capture_utils as cu
from dexnerf.cameras.zed import ZedImageCapture
from dexgrasp.envs.states.grippers.parallel_jaw_gripper import ParallelJawGripper
from ur5py.ur5 import UR5Robot
from dexgrasp.envs.actions import Grasp3D
import multiprocessing as mp
import time
from queue import Empty

class MPWrapper(mp.Process):
    def __init__(self,cls,*args,**kwargs):
        '''
        wraps the given class type specified by cls with the args provided
        '''
        super().__init__()
        self.cls=cls
        #filter out defaults
        self.fn_names = list(filter(lambda x:'__' not in x,dir(self.cls)))
        #define a new attr of this class matching the names, but calling _run_fn
        self.cmd_q = mp.Manager().Queue()#holds commands to execute
        self.resp_q = mp.Manager().Queue()#holds results of commands
        for fn_name in self.fn_names:
            self._add_function(fn_name)
        self.args=args
        self.kwargs=kwargs
        self.daemon=True
        self.start()

    def _add_function(self, fn_name):
        def wrapper_fn(*fn_args,**fn_kwargs):
            #first, enqueue the command
            cmd_spec = (fn_name,fn_args,fn_kwargs)
            self.cmd_q.put(cmd_spec)
            #then, wait for response
            resp = self.resp_q.get(block=True)
            #then return response
            return resp
        setattr(self,fn_name,wrapper_fn)

    def run(self):
        '''
        loops and runs commands and enqueues responses
        '''
        #instantiate the object
        self.obj = self.cls(*self.args,**self.kwargs)
        while True:
            #check if the queue has something
            fn_name,fn_args,fn_kwargs = self.cmd_q.get(block=True)
            #execute the command
            resp = getattr(self.obj,fn_name)(*fn_args,**fn_kwargs)
            self.resp_q.put(resp)

class AsyncCapture(mp.Process):
    def __init__(self, use_stereo:bool, *zed_args, **zed_kwargs):
        '''
        rob can be any object that supports .get_pose() and .get_joints() (ie either UR5RObot ur YuMiArm)
        '''
        super().__init__()
        self.zed_args = zed_args
        self.zed_kwargs = zed_kwargs
        self.img_q = mp.Manager().Queue()#queue which ONLY holds img,pose pairs
        self.cmd_q = mp.Manager().Queue()#queue which holds generic input cmd data (such as last traj joint angles)
        self.res_q = mp.Manager().Queue()#queue which returns generic output data
        self.rob_q = mp.Manager().Queue()#queue for handling commands to robot
        self.trigger_capture = mp.Value('i',0)
        self.cap_threshold=.03#distance between images to save them
        self.use_stereo = use_stereo
        self.daemon = True
        self.rob = MPWrapper(UR5Robot)
        self.start()
        self.zed_intr = self.res_q.get(True)
        self.zed_translation = self.res_q.get(True)

    def run(self):
        self.zed=ZedImageCapture(*self.zed_args,**self.zed_kwargs)
        imgl, imgr = self.zed.capture_image()
        self.res_q.put(self.zed.intrinsics)
        self.res_q.put(self.zed.stereo_translation)
        left_to_right = RigidTransform(translation=self.zed.stereo_translation)
        H = RigidTransform.load('cfg/T_zed_wrist.tf')
        self.rob.set_tcp(H)
        self.rob.start_freedrive()
        while True:
            if self.trigger_capture.value>=1:
                lastpose=None
                last = self.cmd_q.get()
                while True:
                    time.sleep(.01)
                    pose = self.rob.get_pose()
                    # if the norm of the difference of last pose and current pose exceeds cap_threshold, take an image
                    if (lastpose is None or np.linalg.norm(pose.translation - lastpose.translation) > self.cap_threshold):
                        imgl, imgr = self.zed.capture_image()
                        lastpose=pose
                        self.img_q.put((imgl,pose))
                        if self.use_stereo:
                            right_pose = pose*left_to_right.as_frames(pose.from_frame,pose.from_frame)
                            self.img_q.put((imgr,right_pose))
                    #if we reached the last joint angle, end the capture
                    if last is not None and np.linalg.norm(self.rob.get_joints() - last) < 0.2:
                        with self.trigger_capture.get_lock():
                            self.trigger_capture.value = 0
                    if self.trigger_capture.value==0:
                        break
    
    @property
    def intrinsics(self):
        return self.zed_intr
    
    @property 
    def stereo_translation(self):
        return self.zed_translation

    def done(self):
        '''
        returns true if the capture process has finished
        '''
        return self.trigger_capture.value==0
    
    def end_cap(self):
        with self.trigger_capture.get_lock():
            self.trigger_capture.value = 0

    def trigger_cap(self,last_joints):
        '''
        Triggers the capture which will start populating the queue with images and poses
        If last_joints is specified, it will end when the robot reaches that pose, otherwise it will only stop when
        triggered externally
        '''
        self.cmd_q.put(last_joints)
        with self.trigger_capture.get_lock():
            self.trigger_capture.value = 1
    
    def get_imgs(self):
        """
        returns the top images on the queue and removes them
        """
        imgs,poses=[],[]
        while True:
            try:
                im,pose = self.img_q.get(True,timeout=.04)
                imgs.append(im)
                poses.append(pose)
            except Empty:
                break
        return imgs,poses
    
class URCapture:
    def __init__(self):
        self.hardware_proc = AsyncCapture(use_stereo=True,resolution="720p", exposure=60, 
                    gain=70, whitebalance_temp=3000, fps=100)
    
    def start_cap(self):
        self.hardware_proc.trigger_cap(None)
    
    def stop_cap(self):
        self.hardware_proc.end_cap()

    def freedrive_capture(self):
        self.start_cap()
    
    def get_imgs(self):
        return self.hardware_proc.get_imgs()
    

class YuMiCapture:
    L_HOME = np.array(
        [
            -0.62385274,
            -1.24035383,
            0.27775658,
            -0.06119772,
            1.73222464,
            1.71454111,
            0.04887163,
        ]
    )
    R_HOME = np.array(
        [ 0.3992163 , -1.04023631, -0.46078176, -1.14721656, -1.2195347 ,
        1.86848141, -3.88782557]
    )
    R_DROP_JOINTS=np.array([ 0.64155071, -0.86203855, -1.03620003, -1.04685986, -0.48562443,
        1.19153116, -0.97755016])
    R_BEFORE_GRASP=np.array([ 0.47626055, -0.57745249, -0.23186769,  0.70958176, -1.22619979,
        1.14700857, -2.7594285 ])
    def __init__(
        self,
        gripper_tcp: RigidTransform,
        cam_to_wrist_file="cfg/T_zed_wrist.tf",
        speed=(0.15, np.pi / 2),
    ):
        trans = RigidTransform.load(cam_to_wrist_file)
        self.yk = YK()
        self.yk.set_tcp(
            l_tool=trans,
            r_tool=gripper_tcp.as_frames(self.yk.r_tcp_frame, self.yk.base_frame),
        )
        self.yumi = YuMi(l_tcp=trans, r_tcp=gripper_tcp)
        self.zed=AsyncCapture(self.yumi.left,use_stereo=False,resolution="720p", exposure=60, 
                    gain=1, whitebalance_temp=4700, fps=100
        )
        self.planner = Planner()
        self.speed = self.default_speed = speed

    def remove_obj(self,grasp:Grasp3D):
        self.speed = (0.7, np.pi)
        self.home_left()
        self.speed = (0.3, np.pi)
        # home the right hand to get ready for grasping
        # self.go_configs(r_q=[self.R_BEFORE_GRASP])
        self.yumi.right.open_gripper()
        grasp_pose = grasp.pose * self.yk.r_tcp.as_frames(
            grasp.pose.from_frame, grasp.pose.from_frame
        )
        grasp_pose.translation[2]=np.clip(grasp_pose.translation[2],.07,np.inf)
        #compute the pregrasp pose
        # becuase the convention for Grasp3D is to return wrist frame coordinates, we convert to the tcp frame coordinates here
        pregrasp_pose = (grasp_pose * RigidTransform(
            translation=[0, 0, -0.12],
            from_frame=grasp.pose.from_frame,
            to_frame=grasp.pose.from_frame,
        )).as_frames(self.yk.r_tcp_frame,self.yk.base_frame)
        #go to the pregrasp pose
        self.yumi.right.sync()
        _,pregrasp_joints = self.yk.ik(right_pose=pregrasp_pose,right_qinit=self.R_BEFORE_GRASP,solve_type='Manipulation1')
        _,joints=self.yk.ik(right_pose=grasp_pose.as_frames(self.yk.r_tcp_frame,self.yk.base_frame),right_qinit=pregrasp_joints,solve_type='Distance')
        if pregrasp_joints is None or joints is None:
            self.speed=self.default_speed
            return False
        # self.go_configs(r_q=[pregrasp_joints])
        # calculate the downwards pose
        #move down and grasp
        self.yumi.right.move_joints_traj([self.R_BEFORE_GRASP,pregrasp_joints,joints],speed=[(.7,np.pi),(.5,np.pi),(.15,np.pi)],zone='fine')
        # self.go_configs(r_q=[pregrasp_joints,joints])
        self.yumi.right.sync()
        #TODO add this line back if you want it to actually grab
        # self.yumi.right.close_gripper()
        self.yumi.right.sync()
        time.sleep(1)#wait for gripper to close
        #move back up to clear the table
        self.speed=(.4,np.pi)
        self.go_configs(r_q=[pregrasp_joints,self.R_DROP_JOINTS])
        self.yumi.right.sync()
        self.yumi.right.open_gripper()
        self.yumi.right.sync()
        time.sleep(.5)
        self.speed=(.7,np.pi)
        self.home_right()
        self.speed=self.default_speed
        return True
        

    def grasp(self, grasp: Grasp3D):
        """
        executes the grasp motion
        """
        self.speed = (0.3, np.pi)
        self.home_left()
        self.yumi.left.sync()
        # home the right hand to get ready for grasping
        self.go_configs(r_q=[self.R_BEFORE_GRASP])
        self.yumi.right.open_gripper()
        self.yumi.right.sync()
        grasp_pose = grasp.pose * self.yk.r_tcp.as_frames(
            grasp.pose.from_frame, grasp.pose.from_frame
        )
        grasp_pose.translation[2]=np.clip(grasp_pose.translation[2],.07,np.inf)
        # becuase the convention for Grasp3D is to return wrist frame coordinates, we convert to the tcp frame coordinates here
        pregrasp_pose = (grasp_pose * RigidTransform(
            translation=[0, 0, -0.12],
            from_frame=grasp.pose.from_frame,
            to_frame=grasp.pose.from_frame,
        )).as_frames(self.yk.r_tcp_frame,self.yk.base_frame)
        self.speed=(.3,np.pi)
        self.go_pose_plan_single('right',pregrasp_pose)
        self.yumi.right.sync()
        pregrasp_joints=self.yumi.right.get_joints()
        _,joints=self.yk.ik(right_pose=grasp_pose.as_frames(self.yk.r_tcp_frame,self.yk.base_frame),right_qinit=self.yumi.right.get_joints(),solve_type='Distance')
        self.go_configs(r_q=[joints])
        self.yumi.right.close_gripper()
        time.sleep(1.5)
        self.yumi.right.sync()
        self.go_configs(r_q=pregrasp_joints)
        self.sync()
        self.speed=self.default_speed
        self.go_configs(r_q=[joints])
        self.sync()
        self.yumi.right.open_gripper()
        time.sleep(.6)
        self.go_configs(r_q=pregrasp_joints)
        self.sync()

    def home_left(self):
        try:
            self.go_config_plan(l_q=self.L_HOME)
        except:
            self.go_configs(l_q=self.L_HOME)

    def home_right(self):
        try:
            self.go_config_plan(r_q=self.R_HOME)
        except:
            self.go_configs(r_q=self.R_HOME)

    def sync(self):
        self.yumi.left.sync()
        self.yumi.right.sync()

    def go_config_plan(self, l_q=None, r_q=None, table_z=0.03):
        self.sync()
        l_cur = self.yumi.left.get_joints()
        r_cur = self.yumi.right.get_joints()
        l_path, r_path = self.planner.plan(l_cur, r_cur, l_q, r_q, table_z=table_z)
        self.go_configs(l_path, r_path, True)

    def go_configs(self, l_q=[], r_q=[], together=False):
        """
        moves the arms along the given joint trajectories
        l_q and r_q should be given in urdf format as a np array
        """
        if together and len(l_q) == len(r_q):
            self.yumi.move_joints_sync(l_q, r_q, speed=self.speed)
        else:
            if len(l_q) > 0:
                self.yumi.left.move_joints_traj(l_q, speed=self.speed, zone="z10")
            if len(r_q) > 0:
                self.yumi.right.move_joints_traj(r_q, speed=self.speed, zone="z10")

    def go_pose_plan_single(self,which_arm,pose,table_z=.03,mode='Manipulation1'):
        if which_arm=='left':
            self.go_pose_plan(l_target=pose,table_z=table_z,mode=mode)
        else:
            self.go_pose_plan(r_target=pose,table_z=table_z,mode=mode)

    def go_pose_plan(
        self, l_target=None, r_target=None, fine=False, table_z=.03,
                mode='Manipulation1'):
        self.sync()
        l_cur = self.yumi.left.get_joints()
        r_cur = self.yumi.right.get_joints()

        res = self.planner.plan_to_pose(
            l_cur, r_cur, self.yk, l_target, r_target, table_z=table_z,mode=mode
        )
        if res is None:
            raise RuntimeError("Planning to pose failed")
        l_path, r_path = res

        self.go_configs(l_path, r_path, True)
        if fine:
            if l_target is not None:
                self.y.left.goto_pose(l_target, speed=self.speed)
            if r_target is not None:
                self.y.right.goto_pose(r_target, speed=self.speed)
    def move_hemi(
        self,
        R,
        th_N,
        phi_N,
        th_bounds,
        phi_bounds,
        look_at,
        center_pos,
        th_first
    ):
        """
        R: radius of sphere
        theta_N: number of points around the z axis
        phi_N: number of points around the elevation axis
        look_pos: 3D position in world coordinates to point the camera at
        center_pos: 3D position in world coords to center the hemisphere
        th_first: if True, moves around theta axis first (rotation around z)
        """
        self.speed=(.7,np.pi)
        self.home_right()
        self.speed=self.default_speed
        poses = cu._generate_hemi(
            R, th_N, phi_N, th_bounds, phi_bounds, look_at, center_pos, th_first
        )
        traj = self._hemi_to_traj(poses)
        try:
            self.speed=(.5,np.pi)
            self.go_config_plan(l_q=traj[0])
        except:
            self.yumi.left.move_joints_traj(traj[0], speed=(.5,np.pi))
        self.speed=self.default_speed
        self.yumi.left.sync()
        self.yumi.left.move_joints_traj(traj, speed=self.speed, zone="z100")
        return traj[-1]

    def run_capture_hemi(
        self,
        R,
        th_N,
        phi_N,
        th_bounds,
        phi_bounds,
        look_at,
        center_pos,
        th_first,
    ):
        """
        R: radius of sphere
        theta_N: number of points around the z axis
        phi_N: number of points around the elevation axis
        look_pos: 3D position in world coordinates to point the camera at
        center_pos: 3D position in world coords to center the hemisphere
        th_first: if True, moves around theta axis first (rotation around z)
        """
        last = self.move_hemi(R,th_N,phi_N,th_bounds,phi_bounds,look_at,center_pos,th_first)
        imgs, poses = [], []
        self.zed.trigger_cap(last)
        imgs,poses=[],[]
        while True:
            ims,ps = self.zed.get_imgs()
            imgs.extend(ims)
            poses.extend(ps)
            if self.zed.done():
                break
        self.yumi.left.sync()
        return imgs, poses

    def run_capture_joints(self):
        traj=np.load('joint_traj.npy')
        last=traj[-1]
        self.home_right()
        # self.yumi.left.move_joints_traj(traj[0], speed=self.speed)
        # self.yumi.left.sync()
        self.yumi.left.move_joints_traj(traj,speed=self.default_speed,zone='z100')
        self.zed.trigger_cap(last)
        
    def run_capture_hemi_async(self,
        R,
        th_N,
        phi_N,
        th_bounds,
        phi_bounds,
        look_at,
        center_pos,
        th_first
    ):
        last = self.move_hemi(R,th_N,phi_N,th_bounds,phi_bounds,look_at,center_pos,th_first)
        self.zed.trigger_cap(last)

    def _hemi_to_traj(self, poses):
        """
        given the poses returned by _generate_hemi, filter them for reachability and traj smoothness
        returns: list of joints which can be executed on the yumi
        """
        t_tol = 0.03
        r_tol = 0.02
        traj = []
        lastq = np.zeros(7)
        for i, p in enumerate(poses):
            if i == 0:
                joints, _ = self.yk.ik(
                    left_pose=p,
                    left_qinit=lastq,
                    solve_type="Manipulation1",
                    bs=[t_tol, t_tol, t_tol, r_tol, r_tol, np.deg2rad(20)],
                )
            else:
                joints, _ = self.yk.ik(
                    left_pose=p,
                    left_qinit=lastq,
                    solve_type="Distance",
                    bs=[t_tol, t_tol, t_tol, r_tol, r_tol, np.deg2rad(20)],
                )
            if joints is not None:
                lastq = joints
                traj.append(joints)
        return np.array(traj)

    def __del__(self):
        self.zed.kill()
        self.zed.join()
        self.zed.close()