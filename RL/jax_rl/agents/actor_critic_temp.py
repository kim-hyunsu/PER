import flax

from jax_rl.networks.common import Model, PRNGKey


@flax.struct.dataclass
class ActorCriticTemp:
    actor: Model
    critic: Model
    target_critic: Model
    temp: Model
    rng: PRNGKey

    def set_state(self,state):
        if hasattr(self.actor, "set_state"):
            self.actor.set_state(state)
        if hasattr(self.critic, "set_state"):
            self.critic.set_state(state)
    def get_current_state(self):
        current_state = None
        if hasattr(self.actor, "get_current_state"):
            actor_state  = self.actor.get_current_state()
            if actor_state is not None:
                current_state = actor_state
        if hasattr(self.actor, "get_current_state"):
            critic_state = self.critic.get_current_state()
            if critic_state is not None:
                current_state = critic_state
        
        return current_state