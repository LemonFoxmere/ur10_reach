# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Pre-defined configs
##

# from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip
from .ur10 import UR_GRIPPER_CFG


##
# Scene definition
##


@configclass
class Ur10SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # cartpole
    # robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ur10
    robot: ArticulationCfg = UR_GRIPPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=5000.0),
    )

    # table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"  # type: ignore
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_effort = mdp.JointEffortActionCfg(
    #     asset_name="robot", joint_names=["slider_to_cart"], scale=100.0
    # )
    arm_action: ActionTerm = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],  # TODO: try shortening
        scale=1,
        use_default_offset=True,
        debug_vis=True,
    )


@configclass
class CommandsCfg:
    """Command specification for the MDP"""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="ee_link",  # end effector
        resampling_time_range=(4.0, 4.0),  # every 4 seconds (min_time, max_time)
        debug_vis=True,
        # These are essentially ranges of poses that can be commanded for the end of the robot during training
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=(math.pi / 2, math.pi / 2),
            yaw=(-3.14, 3.14),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (with order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        pose_command = ObsTerm(  # this is from our command cfg
            func=mdp.generated_commands, params={"command_name": "ee_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.75, 1.75), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) end effector tracking reward
    ee_pos_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
            "command_name": "ee_pose",
        },
    )

    # (2) end effector tracking (fine grained)
    ee_pos_tracking_fine = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,  # 1 - tanh(d) produces a number close and closer to 1 as we start moving more accurately
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
            "std": 0.1,
            "command_name": "ee_pose",
        },
    )

    # (3) action penalty
    action_rt = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Cart out of bounds
    # cart_out_of_bounds = DoneTerm(
    #     func=mdp.joint_pos_out_of_manual_limit,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
    #         "bounds": (-3.0, 3.0),
    #     },
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # the point of these terms is to slowly ramp up the penalties so that the agent
    # can explore properly at the beginning of training.
    action_rt = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "action_rt",
            "weight": -0.005,
            "num_steps": 4500,
        },  # slowly increase penalty to -0.005 from -0.0001 (-1e-4) by step 4500
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500},
    )


##
# Environment configuration
##


@configclass
class Ur10EnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: Ur10SceneCfg = Ur10SceneCfg(num_envs=2000, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 3.0
        # viewer settings
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1 / 60
        self.sim.render_interval = self.decimation


@configclass
class Ur10EnvCfg_PLAY(Ur10EnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()  # run parent init script

        # smaller scnee
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization
        self.observations.policy.enable_corruption = False
