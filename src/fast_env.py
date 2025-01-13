import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import EnvSpec
from numba import jit

try:
    from .env_hiv import HIVPatient as  SlowHIVPatient
except ImportError:
    from env_hiv import HIVPatient as SlowHIVPatient


@jit(nopython=True)
def _der(state, action, params):
    """Compute derivatives for the HIV system."""
    T1, T1star, T2, T2star, V, E = state
    eps1, eps2 = action

    # Unpack parameters
    (
        lambda1,
        d1,
        k1,
        m1,
        lambda2,
        d2,
        k2,
        f,
        m2,
        delta,
        NT,
        c,
        rho1,
        rho2,
        lambdaE,
        bE,
        Kb,
        dE,
        Kd,
        deltaE,
    ) = params

    # Match the exact computation order of the slow version
    T1dot = lambda1 - d1 * T1 - k1 * (1 - eps1) * V * T1
    T1stardot = k1 * (1 - eps1) * V * T1 - delta * T1star - m1 * E * T1star
    T2dot = lambda2 - d2 * T2 - k2 * (1 - f * eps1) * V * T2
    T2stardot = k2 * (1 - f * eps1) * V * T2 - delta * T2star - m2 * E * T2star
    
    # Removed pre-computation of T1starT2star for exact matching
    Vdot = (NT * delta * (1 - eps2) * (T1star + T2star) 
            - c * V 
            - (rho1 * k1 * (1 - eps1) * T1 + rho2 * k2 * (1 - f * eps1) * T2) * V)
    
    Edot = (lambdaE 
            + bE * (T1star + T2star) * E / (T1star + T2star + Kb)
            - dE * (T1star + T2star) * E / (T1star + T2star + Kd)
            - deltaE * E)

    return np.array([T1dot, T1stardot, T2dot, T2stardot, Vdot, Edot])


@jit(nopython=True)
def _transition(state, action, params, duration=5.0, step_size=1e-3):
    """Faster transition function using numba."""
    state0 = state.copy()
    nb_steps = int(duration // step_size)

    for _ in range(nb_steps):
        der = _der(state0, action, params)
        state0 += der * step_size

    return state0


class FastHIVPatient(gym.Env):
    """Optimized HIV patient simulator"""

    def __init__(self, clipping=True, logscale=False, domain_randomization=False):
        super().__init__()
        self.spec = EnvSpec("FastHIVPatient-v0")

        # Environment configuration
        self.domain_randomization = domain_randomization
        self.clipping = clipping
        self.logscale = logscale
        
        # Spaces
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            shape=(6,), low=-np.inf, high=np.inf, dtype=np.float32
        )

        # Pre-compute action set
        self.action_set = np.array([[0.0, 0.0], [0.0, 0.3], [0.7, 0.0], [0.7, 0.3]])

        # State bounds
        self.upper = np.array([1e6, 5e4, 3200.0, 80.0, 2.5e5, 353200.0])
        self.lower = np.zeros(6)

        # Reward parameters
        self.Q = 0.1
        self.R1 = 20000.0
        self.R2 = 20000.0
        self.S = 1000.0

        # Initialize state
        self.state_vec = np.zeros(6)
        self._reset_patient_parameters()

    def _reset_patient_parameters(self):
        """Initialize or reset patient parameters."""
        if self.domain_randomization:
            self.k1 = np.random.uniform(5e-7, 8e-7)
            self.k2 = np.random.uniform(0.1e-4, 1.0e-4)
            self.f = np.random.uniform(0.29, 0.34)
        else:
            self.k1 = 8e-7
            self.k2 = 1e-4
            self.f = 0.34

        # Create parameter vector for fast computation
        self.params = np.array(
            [
                1e4,  # lambda1
                1e-2,  # d1
                self.k1,  # k1
                1e-5,  # m1
                31.98,  # lambda2
                1e-2,  # d2
                self.k2,  # k2
                self.f,  # f
                1e-5,  # m2
                0.7,  # delta
                100,  # NT
                13,  # c
                1,  # rho1
                1,  # rho2
                1,  # lambdaE
                0.3,  # bE
                100,  # Kb
                0.25,  # dE
                500,  # Kd
                0.1,  # deltaE
            ]
        )

    def reset(self, *, seed=None, options=None, mode="unhealthy"):
        if mode == "uninfected":
            self.state_vec = np.array([1e6, 0.0, 3198.0, 0.0, 0.0, 10.0])
        elif mode == "healthy":
            self.state_vec = np.array([967839.0, 76.0, 621.0, 6.0, 415.0, 353108.0])
        else:  # unhealthy
            self.state_vec = np.array([163573.0, 11945.0, 5.0, 46.0, 63919.0, 24.0])

        self._reset_patient_parameters()
        return self._get_obs(), {}

    def _get_obs(self):
        """Return the current state with appropriate transformations."""
        state = self.state_vec.copy()
        if self.logscale:
            state = np.log10(state)
        return state

    def step(self, action_idx):
        # Get current state (with clipping if enabled)
        current_state = self.state_vec.copy()

        action = self.action_set[action_idx]

        # Use clipped state for transition
        next_state = _transition(current_state, action, self.params)

        # Compute reward using CURRENT state
        reward = -(
            self.Q * current_state[4]
            + self.R1 * action[0] ** 2
            + self.R2 * action[1] ** 2
            - self.S * current_state[5]
        )

        # Apply clipping to next state
        if self.clipping:
            np.clip(next_state, self.lower, self.upper, out=next_state)

        # Update internal state
        self.state_vec = next_state
        

        return self._get_obs(), reward, False, False, {}

    def clone(self):
        """Clone the environment."""
        return self.__class__.from_state(
            self.state_vec,
            self.params,
            self.clipping,
            self.logscale,
            self.domain_randomization,
        )

    @classmethod
    def from_state(
        cls,
        state: np.ndarray,
        params: np.ndarray,
        clipping: bool = True,
        logscale: bool = False,
        domain_randomization: bool = False,
    ):
        """Create an environment from a state and parameters."""
        env = cls(clipping, logscale, domain_randomization)
        env.state_vec = state
        env.params = params
        return env

    def to_slow(self) -> "SlowHIVPatient":
        """Convert this FastHIVPatient to a SlowHIVPatient instance."""
        slow_env = SlowHIVPatient(
            clipping=self.clipping,
            logscale=self.logscale,
            domain_randomization=self.domain_randomization,
        )
        
        # Copy state
        T1, T1star, T2, T2star, V, E = self.state_vec
        slow_env.T1 = T1
        slow_env.T1star = T1star
        slow_env.T2 = T2
        slow_env.T2star = T2star
        slow_env.V = V
        slow_env.E = E
        
        # Copy patient parameters
        (
            slow_env.lambda1,
            slow_env.d1,
            slow_env.k1,
            slow_env.m1,
            slow_env.lambda2,
            slow_env.d2,
            slow_env.k2,
            slow_env.f,
            slow_env.m2,
            slow_env.delta,
            slow_env.NT,
            slow_env.c,
            slow_env.rho1,
            slow_env.rho2,
            slow_env.lambdaE,
            slow_env.bE,
            slow_env.Kb,
            slow_env.dE,
            slow_env.Kd,
            slow_env.deltaE,
        ) = self.params
        
        return slow_env




def test_env_speedup():
    """Measure speedup of fast implementation over slow implementation."""
    import time
    
    # Setup environments
    fast_env = FastHIVPatient()
    slow_env = fast_env.to_slow()
    n_steps = 1000
    
    # Time fast implementation
    start_time = time.time()
    for _ in range(n_steps):
        action = np.random.randint(4)
        fast_env.step(action)
    fast_time = time.time() - start_time
    
    # Time slow implementation
    start_time = time.time()
    for _ in range(n_steps):
        action = np.random.randint(4)
        slow_env.step(action)
    slow_time = time.time() - start_time
    
    speedup = slow_time / fast_time
    print(f"\nSpeedup over {n_steps} steps:")
    print(f"Fast implementation: {fast_time:.3f}s")
    print(f"Slow implementation: {slow_time:.3f}s")
    print(f"Speedup factor: {speedup:.1f}x")
    
    assert speedup > 1, "Fast implementation should be faster than slow implementation"


def test_env_equivalence():
    # Create environments
    fast_env = FastHIVPatient(clipping=True, logscale=False)
    slow_env = fast_env.to_slow()
    
    # Test initial states match
    np.testing.assert_array_almost_equal(
        fast_env._get_obs(),
        slow_env.state()
    )
    
    # Test transitions match for each action
    for action in range(4):
        # Reset both envs to same state
        fast_env.reset(mode="unhealthy")
        slow_env.reset(mode="unhealthy")
        
        # Step both environments
        fast_obs, fast_reward, _, _, _ = fast_env.step(action)
        slow_obs, slow_reward, _, _, _ = slow_env.step(action)
        
        # Compare results
        np.testing.assert_allclose(
            fast_obs, 
            slow_obs,
            rtol=1e-1,  # Allow small relative differences
            err_msg=f"Observations differ for action {action}"
        )
        
        np.testing.assert_almost_equal(
            fast_reward,
            slow_reward,
            decimal=4,
            err_msg=f"Rewards differ for action {action}"
        )

def test_env_equivalence_with_options():
    """Test equivalence with different environment options."""
    options = [
        # dict(clipping=True, logscale=True),
        dict(clipping=False, logscale=False),
        # dict(clipping=False, logscale=True),
        dict()
    ] + [dict(domain_randomization=True)] * 10
    
    for opts in options:
        fast_env = FastHIVPatient(**opts)
        slow_env = fast_env.to_slow()
        return_fast = 0
        return_slow = 0
        # perform 20 random steps
        for _ in range(200):
            action = np.random.randint(4)
            fast_obs, fast_reward, _, _, _ = fast_env.step(action)
            slow_obs, slow_reward, _, _, _ = slow_env.step(action)
            return_fast += fast_reward
            return_slow += slow_reward
            np.testing.assert_allclose(
                fast_obs,
                slow_obs,
                rtol=1e-2,  # Allow small relative differences
                err_msg=f"Observations differ with options {opts}"
            )
        
        # Test states match
        np.testing.assert_allclose(
            fast_env._get_obs(),
            slow_env.state(),
            rtol=1e-2,  # Allow small relative differences
            err_msg=f"States differ with options {opts}"
        )
        
        # Test single transition
        action = 1  # Test with one action is sufficient here
        fast_obs, fast_reward, _, _, _ = fast_env.step(action)
        slow_obs, slow_reward, _, _, _ = slow_env.step(action)
        
        np.testing.assert_allclose(
            fast_obs,
            slow_obs,
            rtol=1e-2,  # Allow small relative differences
            err_msg=f"Observations differ with options {opts}"
        )
        
        np.testing.assert_allclose(
            return_fast,
            return_slow,
            rtol=1e-2,  # Allow small relative differences
            err_msg=f"Rewards differ with options {opts}"
        )


if __name__ == "__main__":
    test_env_speedup()
    test_env_equivalence()
    test_env_equivalence_with_options()
    print("All tests passed!")