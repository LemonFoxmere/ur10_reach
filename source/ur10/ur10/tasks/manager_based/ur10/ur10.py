# articulation config

import isaaclab.sim as sim_utils  # get sim utils
from isaaclab.actuators import ImplicitActuatorCfg  # actuator type
from isaaclab.assets.articulation import ArticulationCfg  # articulation type

UR_GRIPPER_CFG = ArticulationCfg(
    # specify USD loc
    spawn=sim_utils.UsdFileCfg(
        usd_path="/pvc/project-files/scraps/ur10/source/ur10/ur10/tasks/manager_based/ur10/models/ur-with-gripper.usd",
        activate_contact_sensors=False,  # no contact sensors
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={  # check with resp to usd joint names
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={  # actuator configs
        "arm": ImplicitActuatorCfg(  # config all arm actuator
            joint_names_expr=[".*"],
            effort_limit=87.0,
            stiffness=800.0,
            damping=100.0,
        ),
        "gripper": ImplicitActuatorCfg(  # specifically the gripper part
            joint_names_expr=["finger_joint"], stiffness=280, damping=50
        ),
    },
)
