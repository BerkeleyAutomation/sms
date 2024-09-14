import numpy as np
from scipy.spatial.transform import Rotation as R
from autolab_core import RigidTransform
import time
from tracikpy import TracIKSolver

ur5_urdf_filepath = '/home/lifelong/sms/sms/ur5_interface/ur5_interface/urdf/ur5_robot.urdf'
ur5_ik_solver = TracIKSolver(ur5_urdf_filepath, 'base_link', 'tool0')
# Pick in frame a and Place in frame b 
def get_servo_pose(curr_joints,base_to_ee,base_to_frame_a,base_to_frame_b):
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
        new_joints = ur5_ik_solver.ik(new_base_to_ee.matrix,qinit=curr_joints)
        if(new_joints is not None):
            joint_distance = np.linalg.norm(new_joints - curr_joints)
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
    
world_to_ee = np.array([[0.223,0.968,-0.111,0.254],
                        [0.975,-0.220,0.038,-0.639],
                        [0.013,-0.117,-0.993,0.175],
                        [0,0,0,1]])
world_to_ee[:3,:3] = np.array([[ 0.22284484,  0.96848212, -0.11127693],
                               [ 0.97477366, -0.21990525,  0.03818379],
                               [ 0.01250994, -0.11697888, -0.99305561]])

world_to_drill = np.array([[0.456,0.240,-0.857,0.18],
                        [0.879,-0.271,0.391,-0.517],
                        [-0.139,-0.932,-0.334,-0.145],
                        [0,0,0,1]])
world_to_drill[:3,:3] = R.from_quat(np.array([-0.01453482,  0.96936036,  0.12828425, -0.19797438]),scalar_first=True).as_matrix()

world_to_shoebox = np.array([[0.044,0.998,-0.035,-0.172],
                        [-0.998,0.045,0.037,-0.453],
                        [0.039,0.033,0.999,-0.025],
                        [0,0,0,1]])
world_to_shoebox[:3,:3] = R.from_quat(np.array([0.72236517, -0.00127543, -0.02553315, -0.69083183]),scalar_first=True).as_matrix()
ee_to_drill = np.linalg.inv(world_to_ee) @ world_to_drill
desired_distance_from_shoebox_to_drill = 0.3

#Rviz is dumb and has slight rotation error when reflecting this
shoebox_to_desired_servo_frame = np.array([[0.0,-1.0,0.0,0.0],
                                           [-1.0,0.0,0.0,0.0],
                                           [0.0,0.0,-1.0,desired_distance_from_shoebox_to_drill],
                                           [0.0,0.0,0.0,1.0]])

world_to_desired_servo_frame = world_to_shoebox @ shoebox_to_desired_servo_frame
world_to_desired_servo_euler = R.from_matrix(world_to_desired_servo_frame[:3,:3]).as_euler('zyx', degrees=False)
world_to_desired_servo_translation = world_to_desired_servo_frame[:3,3]

world_to_ee_rigid_tf = RigidTransform(rotation=world_to_ee[:3,:3],translation=world_to_ee[:3,3],from_frame='ee',to_frame='world')
world_to_drill_rigid_tf = RigidTransform(rotation=world_to_drill[:3,:3],translation=world_to_drill[:3,3],from_frame='object',to_frame='world')
world_to_desired_servo_frame_rigid_tf = RigidTransform(rotation=world_to_desired_servo_frame[:3,:3],translation=world_to_desired_servo_frame[:3,3],from_frame='object',to_frame='world')
curr_joints = np.array([-1.0215113798724573, -2.170565907155172, -1.4338796774493616, -1.1989854017840784, 1.4985288381576538, 2.3457295894622803])
world_to_desired_ee = get_servo_pose(curr_joints,world_to_ee_rigid_tf,world_to_drill_rigid_tf,world_to_desired_servo_frame_rigid_tf)
print(world_to_desired_ee.matrix)
print("Translation: " + str(world_to_desired_ee.translation))
print("Rotation: " + str(R.from_matrix(world_to_desired_ee.rotation).as_quat()))