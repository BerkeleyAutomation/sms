import numpy as np
from autolab_core import RigidTransform,CameraIntrinsics
from autolab_core.transformations import euler_matrix,euler_from_matrix
from PIL import Image
import json
from scipy.optimize import least_squares, minimize
from typing import List
                 
def estimate_cam2rob(H_chess_cams:List[RigidTransform], H_rob_worlds: List[RigidTransform]):
    '''
    Estimates transform between camera and robot EE frame by doing least-squares fitting,
    Input is a list of Hs from chess frame to camera frame, and from robot frame to world frame.
    Jointly estimates the chessboard location and the camera wrist transformation and returns them
    '''
    def residual(x):
        '''
        x is formatted [12,1]; first x,y,z,rx,ry,rz of chessboard to world, then cam to rob
        '''
        err = 0
        H_chess_world = RigidTransform(translation=x[:3],rotation=euler_matrix(x[3],x[4],x[5])[:3,:3],from_frame='chess',to_frame='world')
        H_cam_rob = RigidTransform(translation = x[6:9],rotation = euler_matrix(x[9],x[10],x[11])[:3,:3],
                                   from_frame='cam',to_frame='rob')
        for H_chess_cam,H_rob_world in zip(H_chess_cams,H_rob_worlds):
            H_chess_world_est = H_rob_world * H_cam_rob * H_chess_cam
            err += np.linalg.norm(H_chess_world.translation-H_chess_world_est.translation)
            rot_diff =H_chess_world.rotation@np.linalg.inv(H_chess_world_est.rotation)
            eul_diff = euler_from_matrix(rot_diff)
            err += np.linalg.norm(eul_diff)
        print(err)
        return err
    x0 = np.zeros(12)
    res = minimize(residual,x0,method='SLSQP')
    print(res)
    if not res.success:
        input("Optimization was not successful, press enter to acknowledge")
    x = res.x
    H_chess_world = RigidTransform(translation=x[:3],rotation=euler_matrix(x[3],x[4],x[5])[:3,:3],from_frame='chess',to_frame='world')
    H_cam_rob = RigidTransform(translation = x[6:9],rotation = euler_matrix(x[9],x[10],x[11])[:3,:3],
                                from_frame='cam',to_frame='rob')
    return H_cam_rob,H_chess_world
    

def point_at(cam_t, obstacle_t, extra_R=np.eye(3)):
    '''
    cam_t: numpy array of 3D position of gripper
    obstacle_t: numpy array of 3D position of location to point camera at
    '''
    dir=obstacle_t-cam_t
    z_axis=dir/np.linalg.norm(dir)
    #change the line below to be negative if the camera is difficult to position
    x_axis_dir = -np.cross(np.array((0,0,1)),z_axis)
    if np.linalg.norm(x_axis_dir)<1e-10:
        x_axis_dir=np.array((0,1,0))
    x_axis=x_axis_dir/np.linalg.norm(x_axis_dir)
    y_axis_dir=np.cross(z_axis,x_axis)
    y_axis=y_axis_dir/np.linalg.norm(y_axis_dir)
    #postmultiply the extra rotation to rotate the camera WRT itself
    R = RigidTransform.rotation_from_axes(x_axis,y_axis,z_axis)@extra_R
    H = RigidTransform(translation=cam_t,rotation=R,from_frame='camera',to_frame='base_link')
    #rotate by extra_R which can specify a rotation for the camera
    return H

def _generate_hemi(R, theta_N, phi_N, th_bounds, phi_bounds, look_pos, center_pos, th_first=True, extra_R = np.eye(3)):
    '''
    R: radius of sphere
    theta_N: number of points around the z axis
    phi_N: number of points around the elevation axis
    look_pos: 3D position in world coordinates to point the camera at
    center_pos: 3D position in world coords to center the hemisphere
    '''
    l_tcp_frame = 'l_tcp'
    base_frame = 'base_link'
    poses=[]
    if th_first:
        for phi_i,phi in enumerate(np.linspace(*phi_bounds,phi_N)):
            ps=[]
            for th_i,th in enumerate(np.linspace(*th_bounds,theta_N)):
                point_x = center_pos[0] + R*np.cos(th)*np.sin(phi)
                point_y = center_pos[1] + R*np.sin(th)*np.sin(phi)
                point_z = center_pos[2] + R*np.cos(phi)
                point =np.array((point_x,point_y,point_z))
                ps.append(point_at(point,look_pos,extra_R = extra_R).as_frames(l_tcp_frame,base_frame))
            #every odd theta, reverse the direction so that the resulting traj is relatively smooth
            if phi_i%2==1:
                ps.reverse()
            poses.extend(ps)
        return poses
    else:
        for th_i,th in enumerate(np.linspace(*th_bounds,theta_N)):
            ps=[]
            for phi in np.linspace(*phi_bounds,phi_N):
                point_x = center_pos[0] + R*np.cos(th)*np.sin(phi)
                point_y = center_pos[1] + R*np.sin(th)*np.sin(phi)
                point_z = center_pos[2] + R*np.cos(phi)
                point =np.array((point_x,point_y,point_z))
                ps.append(point_at(point,look_pos,extra_R = extra_R).as_frames(l_tcp_frame,base_frame))
            #every odd theta, reverse the direction so that the resulting traj is relatively smooth
            if th_i%2==1:
                ps.reverse()
            poses.extend(ps)
        return poses

def _generate_list(positions,look_at):
    '''
    positions: list of positions to center the camera
    look_at: position to point at
    '''
    from yumiplanning.yumi_kinematics import YuMiKinematics as YK
    poses=[]
    for pos in positions:
        pose = point_at(pos,look_at).as_frames(YK.l_tcp_frame,YK.base_frame)
        poses.append(pose)
    return poses

def save_data(imgs,poses,savedir,intr:CameraIntrinsics):
    '''
    takes in a list of numpy arrays and poses and saves a nerf dataset in savedir
    '''
    import os
    os.makedirs(savedir,exist_ok=True)
    data_dict = dict()
    data_dict['frames']=[]
    data_dict['fl_x']=intr.fx
    data_dict['fl_y']=intr.fy
    data_dict['cx']=intr.cx
    data_dict['cy']=intr.cy
    data_dict['h']=imgs[0].shape[0]
    data_dict['w']=imgs[0].shape[1]
    data_dict['aabb_scale']=2
    data_dict['scale']=1.2
    pil_images=[]
    for i,(im,p) in enumerate(zip(imgs,poses)):
        #if RGBA, strip the alpha channel out
        if im.shape[2]==4:im=im[...,:3]
        img=Image.fromarray(im)
        pil_images.append(img)
        img.save(f'{savedir}/img{i}.jpg')
        mat=p.matrix
        mat[:3,1]*=-1
        mat[:3,2]*=-1
        frame = {'file_path':f'img{i}.jpg', 'transform_matrix':mat.tolist()}
        data_dict['frames'].append(frame)
    with open(f"{savedir}/transforms.json",'w') as fp:
        json.dump(data_dict,fp)

def load_data(savedir):
    '''
    loads a nerf dataset from savedir
    '''
    import os
    if not os.path.exists(savedir):
        raise FileNotFoundError(f'{savedir} does not exist')
    with open(f"{savedir}/transforms.json",'r') as fp:
        data_dict = json.load(fp)
    poses=[]
    imgs=[]
    for frame in data_dict['frames']:
        #load the img from frame['file_path'] using PIL Image and convert to numpy array
        img=Image.open(f'{savedir}/{frame["file_path"]}')
        imgs.append(np.array(img))
        mat=np.array(frame['transform_matrix'])
        mat[:3,1]*=-1
        mat[:3,2]*=-1
        poses.append(RigidTransform(*RigidTransform.rotation_and_translation_from_matrix(mat)))
    return imgs,poses