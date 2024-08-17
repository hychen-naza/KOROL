import pickle

env_details = {
    'door': {
        'qpos': {
            'total': 30,
            'hand_Tz': 1,
            'hand_Rxyz': 3,
            'hand_joints': 24,
            'door_hinge': 1,
            'latch': 1
        },
        'action': {
            'total': 28,
            'hand_Tz': 1,
            'hand_Rxyz': 3,
            'hand_joints': 24
        },
        'observation': {
            'total': 39,
            'hand_Rxyz': 3,
            'hand_joints': 24,
            'latch_pos': 1,
            'door_hinge_pos': 1,
            'palm_pos': 3,
            'handle_pos': 3,
            'palm-handle': 3,
            'door_open': 1
        }
    },
    'relocate': {
        'qpos': {
            'total': 36,
            'hand_Txyz': 3,
            'hand_Rxyz': 3,
            'hand_joints': 24,
            'obj_Txyz': 3,
            'obj_Rxyz': 3
        },
        'action': {
            'total': 30,
            'hand_Txyz': 3,
            'hand_Rxyz': 3,
            'hand_joints': 24,
        },
        'observation': {
            'total': 39,
            'hand_Txyz': 3,
            'hand_Rxyz': 3,
            'hand_joints': 24,
            'palm-obj': 3,
            'palm-tar': 3,
            'obj-tar': 3
        }
    },
    'hammer': {
        'qpos': {
            'total': 33,
            'hand_Rxy': 2,
            'hand_joints': 24,
            'nail_dir': 1,
            'hammer_Txyz': 3,
            'hammer_Rxyz': 3
        },
        'action': {
            'total': 26,
            'hand_Rxy': 2,
            'hand_joints': 24
        },
        'observation': {
            'total': 46,
            'hand_Rxy': 2,
            'hand_joints': 24,
            'nail_dir': 1,
            'hammer_Txyz_qvel': 3,
            'hammer_Rxyz_qvel': 3,
            'palm_pos': 3,
            'hammer_pos': 3,
            'hammer_xquat': 3,
            'nail_pos': 3,
            'nail_impact': 1
        }
    },
    'pen': {
        'qpos': {
            'total': 30,
            'hand_joints': 24,
            'pen_Txyz': 3,
            'pen_Rxyz': 3
        },
        'action': {
            'total': 24,
            'hand_joints': 24
        },
        'observation': {
            'total': 45,
            'hand_joints': 24,
            'pen_pos': 3,
            'pen_Txyz_qvel': 3,
            'pen_Rxyz_qvel': 3,
            'pen_orientation': 3,
            'desired_orientation': 3,
            'pen_pos-desired_pos': 3,
            'pen_orien-desired_orien': 3,
        }
    }
}

def get_demo_details():
    for task in env_details.keys():
        print(task.capitalize())
        # demo_file = open(task + '_demos_from_policy.pickle', 'rb')
        demo_file = open('../demonstrations/' + task + '-v0_demos.pickle', 'rb')
        demos_from_policy = pickle.load(demo_file)
        demo_file.close()
        print("Number of demos:", len(demos_from_policy))
        print("Each demo contains:", list(demos_from_policy[0].keys()))
        print("Environment consists of:")
        for state_variable in demos_from_policy[0]['init_state_dict'].keys():
            print('--', state_variable, 'of size', len(demos_from_policy[0]['init_state_dict'][state_variable]))
        print('qpos details', env_details[task]['qpos'])
        print('Number of actions:', len(demos_from_policy[0]['actions']))
        print('Action details:', env_details[task]['action'])
        print('Number of observations:', len(demos_from_policy[0]['observations']))
        print('Observation details:', env_details[task]['observation'])
        print('------------------------------')


if __name__ == '__main__':
    get_demo_details()
