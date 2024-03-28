import torch

#x = torch.load("./FactoryTaskNutBoltScrew.pkl")
#x = torch.load("./path/pick_place_action.pkl")
x = torch.load("./FactoryTaskNutBoltPick_actions_execute.pkl")

print("x")
# nut size = m16
# pos_action_scale: [0.1, 0.1, 0.1]
if task == "FactoryTaskNutBoltPick":
    obs_tensors = [fingertip_midpoint_pos, # 3 dim [x,y,z]
                   fingertip_midpoint_quat, # 4 dim [x,y,z,w]
                   fingertip_midpoint_linvel, # 3 dim
                   fingertip_midpoint_angvel, # 3 dim
                   nut_grasp_pos,  # 3 dim
                   nut_grasp_quat]  # 4 dim


elif task == "FactoryTaskNutBoltPlace":
    obs_tensors = [fingertip_midpoint_pos,
                   fingertip_midpoint_quat,
                   fingertip_midpoint_linvel,
                   fingertip_midpoint_angvel,
                   nut_pos,
                   nut_quat,
                   bolt_pos,
                   bolt_quat]


"""intialState:
    franka_arm_initial_dof_pos: [0.3413, -0.8011, -0.0670, -1.8299,  0.0266,  1.0185,  1.0927] 
    fingertip_midpoint_pos_initial: [0.0, -0.2, 0.6]  # initial position of hand above table (higher table height=0.4, than table 0.2)
    fingertip_midpoint_rot_initial: [3.1416, 0, 3.1416]  # initial rotation of fingertips (Euler) 
    nut_pos_xy_initial: [0.0, -0.3]  # initial XY position of nut on table
    bolt_pos_xy_noise: [0.0, 0.0]

rl:
    pos_action_scale: [0.1, 0.1, 0.1]
    rot_action_scale: [0.1, 0.1, 0.1]

actions:
    pos_actions = actions[:, 0:3]*pos_action_scale
    target_fingertip_midpoint_pos = fingertip_midpoint_pos + pos_actions

    rot_actions = actions[:, 3:6]*rot_action_scale
    angle = torch.norm(rot_actions, p=2, dim=-1)
    axis = rot_actions / angle.unsqueeze(-1)
    rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
            if self.cfg_task.rl.clamp_rot:
                rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
                                               rot_actions_quat,
                                               torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,1))
            self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

    """
