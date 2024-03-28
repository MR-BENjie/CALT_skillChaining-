import torch
import numpy as np
import time
#from robot import Robot

pos_action_scale = np.array([0.1,0.1,0.1])
rot_action_scale = np.array([0.1,0.1,0.1])

def quat_from_angle_axis(angle, axis):
    def normalize(x, eps: float = 1e-9):
        return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([xyz, w], dim=-1))

def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

def transition_action(action, rotation=False):
    pos_actions = action[:, 0:3]
    pos_actions *= pos_action_scale

    # Interpret actions as target rot (axis-angle) displacements
    rot_actions = action[:, 3:6]
    rot_actions *= rot_action_scale
    rot_actions = torch.tensor(rot_actions)
    if rotation:
        rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5
    # Convert to quat and set rot target

    angle = torch.norm(rot_actions, p=2, dim=-1)
    axis = rot_actions / angle.unsqueeze(-1)
    rot_actions_quat = quat_from_angle_axis(angle, axis)

    rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
                                   rot_actions_quat,
                                   torch.tensor([0.0, 0.0, 0.0, 1.0]))

    return pos_actions, rot_actions_quat

def exectue_move_action(robot_control, pos_actions, rot_actions_quats, success=True):
    for pos_action, rot_actions_quat in zip(pos_actions, rot_actions_quats):
        current_pose = robot_control.get_current_pose().pose
        current_pose.position.x += pos_action[0].item()
        current_pose.position.y += pos_action[1].item()
        current_pose.position.z += pos_action[2].item()

        target_orientation = quat_mul(rot_actions_quat,
            torch.tensor([current_pose.orientation.x, current_pose.orientation.y,
                          current_pose.orientation.z, current_pose.orientation.w], dtype=torch.float))
        current_pose.orientation.x = target_orientation[0].item()
        current_pose.orientation.y = target_orientation[1].item()
        current_pose.orientation.z = target_orientation[2].item()
        current_pose.orientation.w = target_orientation[3].item()

        robot_control.set_pose_target(current_pose)
        success &= robot_control.go(wait=True)
    return success

def execute_pick_action(robot, action, tolerance=0.0001):
    success = True
    pos_actions, rot_actions_quats = transition_action(action)  # get rot and pos transition action
    robot_control = robot.gripper_group
    robot_control.set_goal_position_tolerance(tolerance)

    success &= robot.reach_gripper_position(0)
    success &= exectue_move_action(robot_control, pos_actions, rot_actions_quats, success)

    print('Arrive nut pose, perpare for lift nut...')
    success &= robot.reach_gripper_position(0.465)   # grasp nut
    time.sleep(2)
    current_pose = robot_control.get_current_pose().pose
    current_pose.position.y += 0.3*pos_action_scale[2]  # lift distance = 0.3
    robot_control.set_pose_target(current_pose)
    success &= robot_control.go(wait=True)

    return success

def execute_place_action(robot, action, tolerance=0.0001):
    success = True
    pos_actions, rot_actions_quats = transition_action(action)
    robot_control = robot.gripper_group
    robot_control.set_goal_position_tolerance(tolerance)

    success &= exectue_move_action(robot_control, pos_actions, rot_actions_quats, success)

    return success

def execute_screw_action(robot, init_obs, action, tolerance=0.0001):
    success = True
    pos_actions, rot_actions_quats = transition_action(action, rotation=True)
    robot_control = robot.gripper_group

    current_pose = robot_control.get_current_pose().pose
    current_pose.position.x = init_obs[0].item()
    current_pose.position.y = init_obs[1].item()
    current_pose.position.z = init_obs[2].item()
    robot_control.set_pose_target(current_pose)
    success &= robot_control.go(wait=True)

    robot_control.set_goal_position_tolerance(tolerance)

    success &= exectue_move_action(robot_control, pos_actions, rot_actions_quats, success)

    return success
def run():
    data_index = 0
    success = True
    robot = Robot()

    data_pick = torch.load("FactoryTaskNutBoltPick.pkl")
    action = data_pick[data_index]
    success &= execute_pick_action(robot, action)
    print("pick success:",str(success))


    data_place = torch.load("FactoryTaskNutBoltPlace.pkl")
    action = data_place[data_index]
    success &= execute_place_action(robot, action)
    print("place success:", str(success))

    picked_index = 0
    data_screw = torch.load("FactoryTaskNutBoltScrew.pkl")
    picked = data_screw['dones'][picked_index].item()
    init_obs = data_screw['obs'][picked,:].cpu().detach().numpy()
    action = data_screw['action'][:,picked_index,:,:].squeeze()
    success &= execute_screw_action(robot, init_obs, action)
    print("screw success:",str(success))


def test():

    data_pick = torch.load("FactoryTaskNutBoltScrew.pkl")
    picked_index = 0
    picked = data_pick['dones'][picked_index].item()
    init_obs = data_pick['obs'][picked,:].cpu().detach().numpy()
    action = data_pick['action'][:,picked_index,:,:].squeeze()


    pos_actions = action[:, 0:3]
    pos_actions *= pos_action_scale

    # Interpret actions as target rot (axis-angle) displacements
    rot_actions = action[:, 3:6]
    rot_actions *= rot_action_scale
    rot_actions = torch.tensor(rot_actions)

    angle = torch.norm(rot_actions, p=2, dim=-1)
    axis = rot_actions / angle.unsqueeze(-1)
    rot_actions_quat = quat_from_angle_axis(angle, axis)

    rot_actions_quats = torch.where(angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
                                   rot_actions_quat,
                                   torch.tensor([0.0, 0.0, 0.0, 1.0]))

    for pos_action, rot_actions_quat in zip(pos_actions, rot_actions_quats):

        tmp = 1+pos_action[0].item()

        target_orientation = quat_mul(rot_actions_quat, torch.tensor([0.1, 0.2,0.3, 0.5], dtype=torch.float))
        tmp2 = target_orientation[0].item()
        print("tmp")


def process_data(file, step =20):
    datas = torch.load(file)
    datas_list = []
    for data in datas:
        index = 0
        length = data.shape[0]
        data_list = []
        while index<=length:
            if index+step<=length:
                data_list.append(np.sum(data[index:index+step,:],axis=0))
            else:
                data_list.append(np.sum(data[index:,:],axis=0))
            index += step
        data = np.vstack(data_list)
        datas_list.append(data)
    torch.save(datas_list,"proc"+file)


def process_screw_data(file, step=20):
    datas_all = torch.load(file)
    data = datas_all["action"]
    datas_list = []

    index = 0
    length = data.shape[0]
    data_list = []
    while index <= length:
        if index + step <= length:
            data_list.append(np.sum(data[index:index + step, :], axis=0, keepdims=True))
        else:
            data_list.append(np.sum(data[index:, :], axis=0,keepdims=True))
        index += step
    data = np.vstack(data_list)
    datas_all["action"] = data
    torch.save(datas_all, "proc" + file)


if __name__ == "__main__":
    #test()
    files = ["FactoryTaskNutBoltPick.pkl","FactoryTaskNutBoltPlace.pkl"]
    for file in files:
        process_data(file)
    process_screw_data("FactoryTaskNutBoltScrew.pkl")



