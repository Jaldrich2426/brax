import torch

from mrf_swarm.costs.base_costs import DimensionSumCost,BoundCost, BaseCost
from mrf_swarm.costs.obstacle_costs import SignedDistanceMap2DCost, KBendingObstacleCost
from mrf_swarm.costs.trajectory_costs import RunningDeltaCost, TerminalDeltaCost, StateBoundsCost


class InputBoundsCost(BaseCost):
    def __init__(self, bounds, sigma=1.0,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}):

        super().__init__(tensor_kwargs)
        

        self.cost_fn=BoundCost(bounds=bounds, sigma=1.0, tensor_kwargs=tensor_kwargs)

    def cost(self, x: torch.Tensor) -> torch.Tensor:
        """
        - Inputs:
            - x : tensor (..., T, x_dim)
        - Returns:
            - cost : tensor (...,)
        """
        return self.cost_fn(x).sum(-1)  # Sum over the horizon: (..., T) -> (...,)
    
class LowerBoundCost(BaseCost):
    def __init__(self, bounds, sigma=1.0,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}) -> None:
        """
        Inputs:
            - bounds: tensor (2, x_dim) where the first row is the lower bound
                      and the second row is the upper bound.
            - sigma : float, scalar term to scale the impact of this cost
        """
        super().__init__()
        self.bounds = torch.as_tensor(bounds, **tensor_kwargs)
        self.sigma = sigma

    def cost(self, x: torch.Tensor) -> torch.Tensor:
        """
        - Inputs:
            - x : tensor (...,x_dim)
        - Returns:
            - cost : tensor (...,)
        """
        return self.sigma * torch.square((x - self.bounds).clamp(max=0)).sum(-1)

class ZRangeCost(BaseCost):
    def __init__(self, bounds, sigma=1.0,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}):
        super().__init__(tensor_kwargs)

        self.cost_fn=LowerBoundCost(bounds=bounds, sigma=sigma, tensor_kwargs=tensor_kwargs)
    
    def cost(self, x) -> torch.Tensor:
        return self.cost_fn(x).sum(-1)
    
class ZCost(BaseCost):
    def __init__(self, z_max, sigma=1.0,
                 tensor_kwargs={"device": "cpu", "dtype": torch.float32}):
        super().__init__(tensor_kwargs)
        self.sigma=sigma
        self.z_max=z_max
    
    def cost(self, x) -> torch.Tensor:
        return self.sigma*(torch.square(self.z_max-x[...,25])).sum(-1)
    
def make_walking_costs(c_pos_x=0., c_vel_x=0.25,c_q=1.,c_qd=1.,c_q_term=1.,c_qd_term=1., c_u=0.2, c_term_x=6., c_pos_z=0., c_vel_z=0.25,c_term_z=0.,c_torso_angle=6.,dim=2, horizon=1, goal_x=None,goal_z=None, use_terminal_cost=True,
                      c_term_torso_angle=6., 
               tensor_kwargs={"device": "cpu", "dtype": torch.float}):

    
    # c_torso_angle=c_pos_z

    # Cost function for goal (x) stability (z) and control effort
    running_body_cost_Qs=torch.zeros((38,38),**tensor_kwargs)
    running_body_cost_x_bars = torch.zeros(38,**tensor_kwargs)
    terminal_body_cost_Qs = torch.zeros((38,38),**tensor_kwargs)
    terminal_body_cost_x_bars = torch.zeros(38,**tensor_kwargs)
    # print(running_body_cost_Qs.shape)
    # print(running_body_cost_x_bars.shape)

    # joint angles
    running_body_cost_Qs[2,2]=c_torso_angle
    terminal_body_cost_Qs[2,2]=c_torso_angle
    running_body_cost_Qs[3,3]=c_q
    terminal_body_cost_Qs[3,3]=c_q_term
    running_body_cost_Qs[4,4]=c_qd
    terminal_body_cost_Qs[4,4]=c_qd_term
    running_body_cost_Qs[5,5]=c_q
    terminal_body_cost_Qs[5,5]=c_q_term
    running_body_cost_Qs[6,6]=c_qd
    terminal_body_cost_Qs[6,6]=c_qd_term
    running_body_cost_Qs[7,7]=c_q
    terminal_body_cost_Qs[7,7]=c_q_term
    running_body_cost_Qs[8,8]=c_qd
    terminal_body_cost_Qs[8,8]=c_qd_term

    # pos velocities
    running_body_cost_Qs[8,8]=c_vel_x
    running_body_cost_Qs[9,9]=c_vel_z

    # joint velocities
    running_body_cost_Qs[11,11]=c_qd
    running_body_cost_Qs[12,12]=c_qd
    running_body_cost_Qs[13,13]=c_qd
    running_body_cost_Qs[14,14]=c_qd
    running_body_cost_Qs[15,15]=c_qd
    running_body_cost_Qs[16,16]=c_qd
    running_body_cost_Qs[17,17]=c_qd
    terminal_body_cost_Qs[11,11]=c_qd_term
    terminal_body_cost_Qs[12,12]=c_qd_term
    terminal_body_cost_Qs[13,13]=c_qd_term
    terminal_body_cost_Qs[14,14]=c_qd_term
    terminal_body_cost_Qs[15,15]=c_qd_term
    terminal_body_cost_Qs[16,16]=c_qd_term
    terminal_body_cost_Qs[17,17]=c_qd_term

    # x coordinates
    running_body_cost_Qs[18,18]=c_pos_x
    running_body_cost_x_bars[18]=goal_x
    terminal_body_cost_Qs[18,18]=c_term_x
    terminal_body_cost_x_bars[18]=goal_x

    # z coordinates
    # running_body_cost_Qs[25,25]=c_pos_z
    # running_body_cost_x_bars[25]=goal_z 
    # terminal_body_cost_Qs[25,25]=c_term_z
    # terminal_body_cost_x_bars[25]=goal_z
    # inputs
    running_body_cost_Qs[32:,32:]=c_u*torch.eye(6)
    # terminal_body_cost_Qs[32:,32:]=c_u*torch.eye(6)



    # running_body_cost_Qs = (0.*torch.eye(2),c_pos_z*torch.eye(1),0.*torch.eye(6),c_vel_x * torch.eye(1),c_vel_z * torch.eye(1),0.*torch.eye(7),c_pos_x * torch.eye(1), 0. * torch.eye(1),c_u * torch.eye(6))
    # running_body_cost_x_bars = (torch.zeros(9),0. * torch.ones(1),0. * torch.ones(1),torch.zeros(7),goal_x * torch.ones(1), goal_z * torch.ones(1),c_u * torch.ones(6))

    # terminal_body_cost_Qs = (c_term_x* torch.eye(1), 0 * torch.eye(1))
    # terminal_body_cost_x_bars = (goal_x, torch.zeros_like(goal_x))
    # terminal_body_cost_Qs = (0.*torch.eye(2),c_pos_z*torch.eye(1),0.*torch.eye(6),0. * torch.eye(1),0. * torch.eye(1),0.*torch.eye(7),c_term_x * torch.eye(1), 0. * torch.eye(1),0. * torch.eye(6))
    # terminal_body_cost_x_bars = (torch.zeros(9),0. * torch.ones(1),0. * torch.ones(1),torch.zeros(7),goal_x * torch.ones(1), goal_z * torch.ones(1),0. * torch.ones(6))
  
    running_body_cost_fn = RunningDeltaCost(Qs=running_body_cost_Qs, x_bars=running_body_cost_x_bars,
                                       tensor_kwargs=tensor_kwargs)

    terminal_body_cost_fn = TerminalDeltaCost(Qs=terminal_body_cost_Qs, x_bars=terminal_body_cost_x_bars,
                                         tensor_kwargs=tensor_kwargs)

    bounds=torch.zeros(2,38,**tensor_kwargs)
    bounds[0,32:]= -0.9
    bounds[1,32:]= 0.9
    bounds_cost_fn = InputBoundsCost(bounds=bounds, sigma=20.0, tensor_kwargs=tensor_kwargs)
    boundsZ=torch.zeros(38,**tensor_kwargs)
    boundsZ[25]=0.8
    boundsZ[26]=-0.02
    boundsZ[27]=-0.02
    boundsZ[28]=-0.02
    boundsZ[29]=-0.02
    boundsZ[30]=-0.02
    boundsZ[31]=-0.02
    z_cost_fn = ZRangeCost(bounds=boundsZ,sigma=50.0,tensor_kwargs=tensor_kwargs)

    z_target_cost_fn = ZCost(z_max=2.0,sigma=50.0,tensor_kwargs=tensor_kwargs)

    if(use_terminal_cost):
        return (running_body_cost_fn, terminal_body_cost_fn,bounds_cost_fn,z_cost_fn, z_target_cost_fn)
    else:
        return (running_body_cost_fn,bounds_cost_fn,z_cost_fn, z_target_cost_fn)
    

def make_terminal_walking_costs( c_term_x=6.,c_term_z=6.,c_torso_angle=6.,c_q_term=1.,c_qd_term=1.,dim=2, horizon=1, goal_x=None,goal_z=None, use_terminal_cost=True,
               tensor_kwargs={"device": "cpu", "dtype": torch.float}):


    # Cost function for goal (x) stability (z) and control effort

    # terminal_body_cost_Qs = (c_term_x* torch.eye(1), 0 * torch.eye(1))
    # # terminal_body_cost_x_bars = (goal_x, torch.zeros_like(goal_x))
    # terminal_body_cost_Qs = (0.*torch.eye(9),0. * torch.eye(1),0. * torch.eye(1),0.*torch.eye(7),c_term_x * torch.eye(1), c_term_z * torch.eye(1),0. * torch.eye(6))
    # terminal_body_cost_x_bars = (torch.zeros(9),0. * torch.ones(1),0. * torch.ones(1),torch.zeros(7),goal_x * torch.ones(1), goal_z * torch.ones(1),0. * torch.ones(6))

    # terminal_body_cost_fn = TerminalDeltaCost(Qs=terminal_body_cost_Qs, x_bars=terminal_body_cost_x_bars,
    #                                      tensor_kwargs=tensor_kwargs)


    # bounds=torch.zeros(2,26,**tensor_kwargs)
    # bounds[0,20:]= -1.0
    # bounds[1,20:]= 1.0
    # bounds_cost_fn = InputBoundsCost(bounds=bounds, sigma=1.0, tensor_kwargs=tensor_kwargs)

    # boundsZ=torch.zeros(26,**tensor_kwargs)
    # boundsZ[1]=1.0
    # z_cost_fn = ZRangeCost(bounds=boundsZ,sigma=10.0,tensor_kwargs=tensor_kwargs)


    # c_torso_angle=

    # Cost function for goal (x) stability (z) and control effort

    terminal_body_cost_Qs = torch.zeros(38,38,**tensor_kwargs)
    terminal_body_cost_x_bars = torch.zeros(38,**tensor_kwargs)

    # joint angles
    terminal_body_cost_Qs[2,2]=c_torso_angle
    terminal_body_cost_Qs[3,3]=c_q_term
    terminal_body_cost_Qs[4,4]=c_qd_term
    terminal_body_cost_Qs[5,5]=c_q_term
    terminal_body_cost_Qs[6,6]=c_qd_term
    terminal_body_cost_Qs[7,7]=c_q_term
    terminal_body_cost_Qs[8,8]=c_qd_term
    
    # pos velocities


    # joint velocities
    terminal_body_cost_Qs[11,11]=c_qd_term
    terminal_body_cost_Qs[12,12]=c_qd_term
    terminal_body_cost_Qs[13,13]=c_qd_term
    terminal_body_cost_Qs[14,14]=c_qd_term
    terminal_body_cost_Qs[15,15]=c_qd_term
    terminal_body_cost_Qs[16,16]=c_qd_term
    terminal_body_cost_Qs[17,17]=c_qd_term

    # x coordinates

    terminal_body_cost_Qs[18,18]=c_term_x
    terminal_body_cost_x_bars[18]=goal_x

    # z coordinates

    # terminal_body_cost_Qs[25,25]=c_term_z
    # terminal_body_cost_x_bars[25]=goal_z
    # inputs

    # terminal_body_cost_Qs[32:,32:]=c_u*torch.eye(6)



    # running_body_cost_Qs = (0.*torch.eye(2),c_pos_z*torch.eye(1),0.*torch.eye(6),c_vel_x * torch.eye(1),c_vel_z * torch.eye(1),0.*torch.eye(7),c_pos_x * torch.eye(1), 0. * torch.eye(1),c_u * torch.eye(6))
    # running_body_cost_x_bars = (torch.zeros(9),0. * torch.ones(1),0. * torch.ones(1),torch.zeros(7),goal_x * torch.ones(1), goal_z * torch.ones(1),c_u * torch.ones(6))

    # terminal_body_cost_Qs = (c_term_x* torch.eye(1), 0 * torch.eye(1))
    # terminal_body_cost_x_bars = (goal_x, torch.zeros_like(goal_x))
    # terminal_body_cost_Qs = (0.*torch.eye(2),c_pos_z*torch.eye(1),0.*torch.eye(6),0. * torch.eye(1),0. * torch.eye(1),0.*torch.eye(7),c_term_x * torch.eye(1), 0. * torch.eye(1),0. * torch.eye(6))
    # terminal_body_cost_x_bars = (torch.zeros(9),0. * torch.ones(1),0. * torch.ones(1),torch.zeros(7),goal_x * torch.ones(1), goal_z * torch.ones(1),0. * torch.ones(6))
  
    terminal_body_cost_fn = TerminalDeltaCost(Qs=terminal_body_cost_Qs, x_bars=terminal_body_cost_x_bars,
                                         tensor_kwargs=tensor_kwargs)

    bounds=torch.zeros(2,38,**tensor_kwargs)
    bounds[0,32:]= -1.0
    bounds[1,32:]= 1.0
    # bounds_cost_fn = InputBoundsCost(bounds=bounds, sigma=1.0, tensor_kwargs=tensor_kwargs)
    boundsZ=torch.zeros(38,**tensor_kwargs)
    boundsZ[25]=1.0
    z_cost_fn = ZRangeCost(bounds=boundsZ,sigma=100.0,tensor_kwargs=tensor_kwargs)
    z_target_cost_fn = ZCost(z_max=2.0,sigma=100.0,tensor_kwargs=tensor_kwargs)

    return (terminal_body_cost_fn,z_cost_fn,z_target_cost_fn)
