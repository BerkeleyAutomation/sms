import time
import numpy as np
from autolab_core import RigidTransform
from tracikpy import TracIKSolver

ur5_urdf_filepath = '/home/lifelong/sms/sms/ur5_interface/ur5_interface/urdf/ur5_robot.urdf'
ur5_ik_solver = TracIKSolver(ur5_urdf_filepath, 'world', 'tool0')

def get_servo_pose(curr_joints,base_to_ee,base_to_frame_a,base_to_frame_b):
    print("IN METHOD")
    start_time = time.time()
    thetas = np.linspace(-np.pi/2,np.pi/2,15)
    base_to_frame_b_variations = []
    for theta in thetas:
        rotation_tf = RigidTransform(rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]),translation=np.zeros(3),to_frame='object',from_frame='object')
        base_to_frame_b_rotation = base_to_frame_b * rotation_tf
        base_to_frame_b_variations.append(base_to_frame_b_rotation)
    min_joint_distance = 1000
    min_base_to_frame_b = None
    
    for base_to_frame_b in base_to_frame_b_variations:
        ee_to_object = base_to_ee.inverse() * base_to_frame_a
        new_base_to_ee = base_to_frame_b * ee_to_object.inverse()
        new_joints = ur5_ik_solver.ik(new_base_to_ee.matrix,qinit=curr_joints,bx=1e-3,by=1e-3,bz=1e-3,brx=1e-2,bry=1e-2,brz=1e-2)
        import pdb
        pdb.set_trace()
        print(new_joints)
        if(new_joints is not None):
            joint_distance = np.linalg.norm(new_joints - curr_joints)
            print("Joint distance: " + str(joint_distance))
            if(joint_distance < min_joint_distance):
                min_joint_distance = joint_distance
                min_base_to_frame_b = base_to_frame_b
    base_to_frame_b = min_base_to_frame_b
    ee_to_object = base_to_ee.inverse() * base_to_frame_a
    new_base_to_ee = base_to_frame_b * ee_to_object.inverse()
    end_time = time.time()
    print("Min joint distance: " + str(min_joint_distance))
    print("Time taken: " + str(end_time - start_time))
    return new_base_to_ee

curr_joints = np.array([-0.9955533186541956, -2.1214593092547815, -1.5139878431903284, -1.2548497358905237, 1.440382480621338, 2.372809886932373])
world_to_ee_rigid_tf = RigidTransform(rotation=np.array([[0.21813594 , 0.95377389 ,-0.20671737],[0.97549109, -0.20682541,  0.07510247],[0.02887637, -0.2180335 , -0.97551399]]),translation=np.array([0.2493659 , -0.60875523 , 0.168769]),from_frame='ee',to_frame='world')
world_to_drill_rigid_tf = RigidTransform(rotation=np.array([[0.908195 ,  -0.05657232 ,-0.41470643],[-0.02336418, -0.99613078 , 0.08472065],[-0.41789468, -0.06725359 ,-0.90600275]]),translation=np.array([0.15664936, -0.58515172, -0.13321406]),from_frame='object',to_frame='world')
world_to_desired_servo_frame_rigid_tf = RigidTransform(rotation=np.array([[0.05657232, -0.908195,    0.41470643],[0.99613078 , 0.02336418, -0.08472065],[0.06725359,  0.41789468,  0.90600275]]),translation=np.array([-0.24068244, -0.51755037,  0.27099241]),from_frame='object',to_frame='world')
world_to_desired_ee = get_servo_pose(curr_joints,world_to_ee_rigid_tf,world_to_drill_rigid_tf,world_to_desired_servo_frame_rigid_tf)