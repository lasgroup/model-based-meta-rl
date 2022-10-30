import numpy as np


class RandomMujocoEnv:

    def __init__(self, random_scale_limit, rand_params):
        self.random_scale_limit = random_scale_limit
        self.rand_params = rand_params
        self.save_parameters()

    def sample_tasks(self, n_tasks):
        """
        Generates randomized parameter sets for the mujoco env

        Args:
            n_tasks (int) : number of different meta-tasks needed

        Returns:
            tasks (list) : an (n_tasks) length list of tasks
        """
        param_sets = []

        for _ in range(n_tasks):
            # body mass -> one multiplier for all body parts

            new_params = {}

            if 'body_mass' in self.rand_params:
                body_mass_multiplyers = np.array(1.5) ** np.random.uniform(-self.random_scale_limit, self.random_scale_limit,  size=self.model.body_mass.shape)
                new_params['body_mass'] = self.init_params['body_mass'] * body_mass_multiplyers

            # body_inertia
            if 'body_inertia' in self.rand_params:
                body_inertia_multiplyers = np.array(1.5) ** np.random.uniform(-self.random_scale_limit, self.random_scale_limit,  size=self.model.body_inertia.shape)
                new_params['body_inertia'] = body_inertia_multiplyers * self.init_params['body_inertia']

            # damping -> different multiplier for different dofs/joints
            if 'dof_damping' in self.rand_params:
                dof_damping_multipliers = np.array(1.3) ** np.random.uniform(-self.random_scale_limit, self.random_scale_limit, size=self.model.dof_damping.shape)
                new_params['dof_damping'] = np.multiply(self.init_params['dof_damping'], dof_damping_multipliers)

            # friction at the body components
            if 'geom_friction' in self.rand_params:
                dof_damping_multipliers = np.array(1.5) ** np.random.uniform(-self.random_scale_limit, self.random_scale_limit, size=self.model.geom_friction.shape)
                new_params['geom_friction'] = np.multiply(self.init_params['geom_friction'], dof_damping_multipliers)

            param_sets.append(new_params)

        return param_sets

    def set_task(self, task):
        for param, param_val in task.items():
            param_variable = getattr(self.model, param)
            assert param_variable.shape == param_val.shape, 'shapes of new parameter value and old one must match'
            if isinstance(getattr(self.model, param), np.ndarray):
                np.copyto(getattr(self.model, param), param_val)
            else:
                setattr(self.model, param, param_val)
        self.cur_params = task

    def get_task(self):
        return self.cur_params

    def save_parameters(self):
        self.init_params = {}
        if 'body_mass' in self.rand_params:
            self.init_params['body_mass'] = self.model.body_mass

        # body_inertia
        if 'body_inertia' in self.rand_params:
            self.init_params['body_inertia'] = self.model.body_inertia

        # damping -> different multiplier for different dofs/joints
        if 'dof_damping' in self.rand_params:
            self.init_params['dof_damping'] = self.model.dof_damping

        # friction at the body components
        if 'geom_friction' in self.rand_params:
            self.init_params['geom_friction'] = self.model.geom_friction
        self.cur_params = self.init_params
