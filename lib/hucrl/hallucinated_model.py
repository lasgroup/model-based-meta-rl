import torch
from rllib.model.transformed_model import TransformedModel


class HallucinatedModel(TransformedModel):

    def __init__(
        self, base_model, transformations, beta=1.0, hallucinate_rewards=False
    ):
        super().__init__(base_model, transformations)
        self.a_dim_action = base_model.dim_action
        self.dim_action = (self.dim_action[0] + self.dim_state[0],)
        self.h_dim_action = self.dim_state
        self.beta = beta

    def forward(self, state, action, next_state=None):
        """Get Optimistic Next state."""
        dim_action, dim_state = self.a_dim_action[0], self.dim_state[0]
        control_action = action[..., :dim_action]

        optimism_vars = action[..., dim_action : dim_action + dim_state]
        optimism_vars = torch.clamp(optimism_vars, -1.0, 1.0)

        mean, tril = self.predict(state, control_action)
        if torch.all(tril == 0.0):
            return mean

        if optimism_vars.shape[-1] == 0:
            return mean, tril

        return (
            mean + self.beta * (tril @ optimism_vars.unsqueeze(-1)).squeeze(-1),
            torch.zeros_like(tril),
        )

    def scale(self, state, action):
        """Get scale at current state-action pair."""
        control_action = action[..., : self.a_dim_action[0]]
        scale = super().scale(state, control_action)

        return scale
