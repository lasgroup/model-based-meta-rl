from abc import ABC, abstractmethod
from typing import NamedTuple, Union, Optional, Any
import numpy as np


def _scan(f, init, xs, length=None):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, np.stack(ys)


class CarParams(NamedTuple):
    """
    d_f, d_r : Represent grip of the car. Range: [0.015, 0.025]
    b_f, b_r: Slope of the pacejka. Range: [2.0 - 4.0].

    delta_limit: [0.3 - 0.5] -> Limit of the steering angle.

    c_m_1: Motor parameter. Range [0.2, 0.5]
    c_m_1: Motor friction, Range [0.00, 0.007]
    c_f, c_r: [1.0 2.0] # motor parameters: source https://web.mit.edu/drela/Public/web/qprop/motor1_theory.pdf,
    https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/Embedded
    -Control-Systems/LectureNotes/6_Motor_Control.pdf # or look at:
    https://video.ethz.ch/lectures/d-mavt/2021/spring/151-0593-00L/00718f4f-116b-4645-91da-b9482164a3c7.html :
    lecture 2 part 2
    c_m_1: max current of motor: [0.2 - 0.5] c_m_2: motor resistance due to shaft: [0.01 - 0.15]
    """
    m: Union[np.array, float] = np.array(0.05)  # [0.04, 0.08]
    i_com: Union[np.array, float] = np.array(27.8e-6)  # [1e-6, 5e-6]
    l_f: Union[np.array, float] = np.array(0.03)  # [0.025, 0.05]
    l_r: Union[np.array, float] = np.array(0.035)  # [0.025, 0.05]
    g: Union[np.array, float] = np.array(9.81)
    d_f: Union[np.array, float] = np.array(0.02)  # [0.015, 0.025]
    c_f: Union[np.array, float] = np.array(1.2)  # [1.0, 2.0]
    b_f: Union[np.array, float] = np.array(2.58)  # [2.0, 4.0]
    d_r: Union[np.array, float] = np.array(0.017)  # [0.015, 0.025]
    c_r: Union[np.array, float] = np.array(1.27)  # [1.0, 2.0]
    b_r: Union[np.array, float] = np.array(3.39)  # [2.0, 4.0]
    c_m_1: Union[np.array, float] = np.array(0.2)  # [0.2, 0.5]
    c_m_2: Union[np.array, float] = np.array(0.05)  # [0.00, 0.007]
    c_d: Union[np.array, float] = np.array(0.052)  # [0.01, 0.1]
    steering_limit: Union[np.array, float] = np.array(0.35)
    use_blend: Union[np.array, float] = np.array(
        0.0)  # 0.0 -> no blend (only kinematics), 1.0 -> (kinematics + dynamics)

    # parameters used to compute the blend ratio characteristics
    blend_ratio_ub: Union[np.array, float] = np.array([0.5477225575])
    blend_ratio_lb: Union[np.array, float] = np.array([0.4472135955])
    angle_offset: Union[np.array, float] = np.array([0.0])


class DynamicsModel(ABC):
    def __init__(self,
                 dt: float,
                 x_dim: int,
                 u_dim: int,
                 params: Any,
                 angle_idx: Optional[Union[int, np.array]] = None,
                 dt_integration: float = 0.01,
                 ):
        self.dt = dt
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.params = params
        self.angle_idx = angle_idx

        self.dt_integration = dt_integration
        assert dt >= dt_integration
        assert (dt / dt_integration - int(dt / dt_integration)) < 1e-4, 'dt must be multiple of dt_integration'
        self._num_steps_integrate = int(dt / dt_integration)

    def next_step(self, x: np.array, u: np.array, params: Any) -> np.array:
        def body(carry, _):
            q = carry + self.dt_integration * self.ode(carry, u, params)
            return q, None

        next_state, _ = _scan(body, x, xs=None, length=self._num_steps_integrate)
        if self.angle_idx is not None:
            theta = next_state[self.angle_idx]
            # sin_theta, cos_theta = np.sin(theta), np.cos(theta)
            # next_state = next_state.at[self.angle_idx].set(np.arctan2(sin_theta, cos_theta))
        return next_state

    def ode(self, x: np.array, u: np.array, params) -> np.array:
        assert x.shape[-1] == self.x_dim and u.shape[-1] == self.u_dim
        return self._ode(x, u, params)

    @abstractmethod
    def _ode(self, x: np.array, u: np.array, params) -> np.array:
        pass


class RaceCar(DynamicsModel):
    """
    local_coordinates: bool
        Used to indicate if local or global coordinates shall be used.
        If local, the state x is
            x = [0, 0, theta, vel_r, vel_t, angular_velocity_z]
        else:
            x = [x, y, theta, vel_x, vel_y, angular_velocity_z]
    u = [steering_angle, throttle]
    encode_angle: bool
        Encodes angle to sin and cos if true

    """

    def __init__(self, dt, encode_angle: bool = True, local_coordinates: bool = False, rk_integrator: bool = True):
        super().__init__(dt=dt, x_dim=6, u_dim=2, params=CarParams(), angle_idx=2,
                         dt_integration=1 / 90.)
        self.encode_angle = encode_angle
        self.local_coordinates = local_coordinates
        self.angle_idx = 2
        self.velocity_start_idx = 4 if self.encode_angle else 3
        self.velocity_end_idx = 5 if self.encode_angle else 4
        self.rk_integrator = rk_integrator

    def rk_integration(self, x: np.array, u: np.array, params: CarParams) -> np.array:
        integration_factors = np.asarray([self.dt_integration / 2.,
                                           self.dt_integration / 2., self.dt_integration,
                                           self.dt_integration])
        integration_weights = np.asarray([self.dt_integration / 6.,
                                           self.dt_integration / 3., self.dt_integration / 3.0,
                                           self.dt_integration / 6.0])

        def body(carry, _):
            """one step of rk integration.
            k_0 = self.ode(x, u)
            k_1 = self.ode(x + self.dt_integration / 2. * k_0, u)
            k_2 = self.ode(x + self.dt_integration / 2. * k_1, u)
            k_3 = self.ode(x + self.dt_integration * k_2, u)

            x_next = x + self.dt_integration * (k_0 / 6. + k_1 / 3. + k_2 / 3. + k_3 / 6.)
            """

            def rk_integrate(carry, ins):
                k = self.ode(carry, u, params)
                carry = carry + k * ins
                outs = k
                return carry, outs

            _, dxs = _scan(rk_integrate, carry, xs=integration_factors, length=4)
            dx = (dxs.T * integration_weights).sum(axis=-1)
            q = carry + dx
            return q, None

        next_state, _ = _scan(body, x, xs=None, length=self._num_steps_integrate)
        if self.angle_idx is not None:
            theta = next_state[self.angle_idx]
            # sin_theta, cos_theta = np.sin(theta), np.cos(theta)
            # next_state[self.angle_idx] = np.arctan2(sin_theta, cos_theta)
        return next_state

    def next_step(self, x: np.array, u: np.array, params: CarParams) -> np.array:
        theta_x = np.arctan2(x[..., self.angle_idx], x[..., self.angle_idx + 1]) if self.encode_angle else \
            x[..., self.angle_idx]
        offset = np.clip(params.angle_offset, -np.pi, np.pi)
        theta_x = theta_x + offset
        if not self.local_coordinates:
            # rotate velocity to local frame to compute dx
            velocity_global = x[..., self.velocity_start_idx: self.velocity_end_idx + 1]
            rotated_vel = self.rotate_vector(velocity_global, -theta_x)
            x[..., self.velocity_start_idx: self.velocity_end_idx + 1] = rotated_vel
        if self.encode_angle:
            theta = np.arctan2(x[..., self.angle_idx], x[..., self.angle_idx + 1])

            x_reduced = np.concatenate([x[..., 0:self.angle_idx], np.atleast_1d(theta),
                                         x[..., self.velocity_start_idx:]],
                                        axis=-1)
            if self.rk_integrator:
                x_reduced = self.rk_integration(x_reduced, u, params)
            else:
                x_reduced = super().next_step(x_reduced, u, params)
            next_theta = np.atleast_1d(x_reduced[..., self.angle_idx])
            next_x = np.concatenate([x_reduced[..., 0:self.angle_idx], np.sin(next_theta), np.cos(next_theta),
                                      x_reduced[..., self.angle_idx + 1:]], axis=-1)
        else:
            if self.rk_integrator:
                next_x = self.rk_integration(x, u, params)
            else:
                next_x = super().next_step(x, u, params)

        if self.local_coordinates:
            # convert position to local frame
            pos = next_x[..., 0:self.angle_idx] - x[..., 0:self.angle_idx]
            rotated_pos = self.rotate_vector(pos, -theta_x)
            next_x[..., 0:self.angle_idx] = rotated_pos
        else:
            # convert velocity to global frame
            new_theta_x = np.arctan2(next_x[..., self.angle_idx], next_x[..., self.angle_idx + 1]) \
                if self.encode_angle else next_x[..., self.angle_idx]
            new_theta_x = new_theta_x + offset
            velocity = next_x[..., self.velocity_start_idx: self.velocity_end_idx + 1]
            rotated_vel = self.rotate_vector(velocity, new_theta_x)
            next_x[..., self.velocity_start_idx: self.velocity_end_idx + 1] = rotated_vel

        return next_x

    @staticmethod
    def rotate_vector(v, theta):
        v_x, v_y = v[..., 0], v[..., 1]
        rot_x = v_x * np.cos(theta) - v_y * np.sin(theta)
        rot_y = v_x * np.sin(theta) + v_y * np.cos(theta)
        return np.concatenate([np.atleast_1d(rot_x), np.atleast_1d(rot_y)], axis=-1)

    @staticmethod
    def _accelerations(x, u, params: CarParams):
        """Compute acceleration forces for dynamic model.
        Inputs
        -------
        x: np.ndarray,
            shape = (6, ) -> [x, y, theta, velocity_r, velocity_t, angular_velocity_z]
        u: np.ndarray,
            shape = (2, ) -> [steering_angle, throttle]

        Output
        ------
        acceleration: np.ndarray,
            shape = (3, ) -> [a_r, a_t, a_theta]
        """
        i_com = params.i_com
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        m = params.m
        l_f = params.l_f
        l_r = params.l_r
        d_f = params.d_f * params.g
        d_r = params.d_r * params.g
        c_f = params.c_f
        c_r = params.c_r
        b_f = params.b_f
        b_r = params.b_r
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2

        c_d = params.c_d

        delta, d = u[0], u[1]

        alpha_f = - np.arctan(
            (w * l_f + v_y) /
            (v_x + 1e-6)
        ) + delta
        alpha_r = np.arctan(
            (w * l_r - v_y) /
            (v_x + 1e-6)
        )
        f_f_y = d_f * np.sin(c_f * np.arctan(b_f * alpha_f))
        f_r_y = d_r * np.sin(c_r * np.arctan(b_r * alpha_r))
        f_r_x = (c_m_1 * d - (c_m_2 ** 2) * v_x - (c_d ** 2) * (v_x * np.abs(v_x)))

        v_x_dot = (f_r_x - f_f_y * np.sin(delta) + m * v_y * w) / m
        v_y_dot = (f_r_y + f_f_y * np.cos(delta) - m * v_x * w) / m
        w_dot = (f_f_y * l_f * np.cos(delta) - f_r_y * l_r) / i_com

        acceleration = np.array([v_x_dot, v_y_dot, w_dot])
        return acceleration

    def _ode_dyn(self, x, u, params: CarParams):
        """Compute derivative using dynamic model.
        Inputs
        -------
        x: np.ndarray,
            shape = (6, ) -> [x, y, theta, velocity_r, velocity_t, angular_velocity_z]
        u: np.ndarray,
            shape = (2, ) -> [steering_angle, throttle]

        Output
        ------
        x_dot: np.ndarray,
            shape = (6, ) -> time derivative of x

        """
        # state = [p_x, p_y, theta, v_x, v_y, w]. Velocities are in local coordinate frame.
        # Inputs: [\delta, d] -> \delta steering angle and d duty cycle of the electric motor.
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        p_x_dot = v_x * np.cos(theta) - v_y * np.sin(theta)
        p_y_dot = v_x * np.sin(theta) + v_y * np.cos(theta)
        theta_dot = w
        p_x_dot = np.array([p_x_dot, p_y_dot, theta_dot])

        accelerations = self._accelerations(x, u, params)

        x_dot = np.concatenate([p_x_dot, accelerations], axis=-1)
        return x_dot

    def _compute_dx_kin(self, x, u, params: CarParams):
        """Compute kinematics derivative for localized state.
        Inputs
        -----
        x: np.ndarray,
            shape = (6, ) -> [x, y, theta, v_x, v_y, w], velocities in local frame
        u: np.ndarray,
            shape = (2, ) -> [steering_angle, throttle]

        Output
        ------
        dx_kin: np.ndarray,
            shape = (6, ) -> derivative of x

        Assumption: \dot{\delta} = 0.
        """
        p_x, p_y, theta, v_x, v_y, w = x[0], x[1], x[2], x[3], x[4], x[5]  # progress
        m = params.m
        l_f = params.l_f
        l_r = params.l_r
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2
        c_d = params.c_d
        delta, d = u[0], u[1]
        v_r = v_x
        v_r_dot = (c_m_1 * d - (c_m_2 ** 2) * v_r - (c_d ** 2) * (v_r * np.abs(v_r))) / m
        beta = np.arctan(np.tan(delta) * 1 / (l_r + l_f))
        v_x_dot = v_r_dot * np.cos(beta)
        # Determine accelerations from the kinematic model using FD.
        v_y_dot = (v_r * np.sin(beta) * l_r - v_y) / self.dt_integration
        # v_x_dot = (v_r_dot + v_y * w)
        # v_y_dot = - v_x * w
        w_dot = (np.sin(beta) * v_r - w) / self.dt_integration
        p_g_x_dot = v_x * np.cos(theta) - v_y * np.sin(theta)
        p_g_y_dot = v_x * np.sin(theta) + v_y * np.cos(theta)
        dx_kin = np.asarray([p_g_x_dot, p_g_y_dot, w, v_x_dot, v_y_dot, w_dot])
        return dx_kin

    def _compute_dx(self, x, u, params: CarParams):
        """Calculate time derivative of state.
        Inputs:
        ------
        x: np.ndarray,
            shape = (6, ) -> [x, y, theta, vel_r, vel_t, angular_velocity_z]
        u: np.ndarray,
            shape = (2, ) -> [steering_angle, throttle]
        params: CarParams,

        Output:
        -------
        dx: np.ndarray, derivative of x


        If params.use_blend <= 0.5 --> only kinematic model is used, else a blend between nonlinear model
        and kinematic is used.
        """
        use_kin = params.use_blend <= 0.5
        v_x = x[3]
        blend_ratio_ub = np.square(params.blend_ratio_ub)
        blend_ratio_lb = np.square(params.blend_ratio_lb)
        blend_ratio = (v_x - blend_ratio_ub) / (blend_ratio_lb + 1E-6)
        blend_ratio = blend_ratio.squeeze()
        lambda_blend = np.min(np.asarray([
            np.max(np.asarray([blend_ratio, 0])), 1])
        )
        dx_kin_full = self._compute_dx_kin(x, u, params)
        dx_dyn = self._ode_dyn(x=x, u=u, params=params)
        dx_blend = lambda_blend * dx_dyn + (1 - lambda_blend) * dx_kin_full
        dx = (1 - use_kin) * dx_blend + use_kin * dx_kin_full
        return dx

    def _ode(self, x, u, params: CarParams):
        """
        Using kinematic model with blending: https://arxiv.org/pdf/1905.05150.pdf
        Code based on: https://github.com/alexliniger/gym-racecar/

        Inputs:
        ------
        x: np.ndarray,
            shape = (6, ) -> [x, y, theta, vel_r, vel_t, angular_velocity_z]
        u: np.ndarray,
            shape = (2, ) -> [steering_angle, throttle]
        params: CarParams,

        Output:
        -------
        dx: np.ndarray, derivative of x
        """
        delta, d = u[0], u[1]
        delta = np.clip(delta, a_min=-1, a_max=1) * params.steering_limit
        d = np.clip(d, a_min=-1., a_max=1)  # throttle
        u = np.array([delta, d])
        dx = self._compute_dx(x, u, params)
        return dx
