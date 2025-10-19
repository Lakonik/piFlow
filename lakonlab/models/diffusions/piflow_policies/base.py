from abc import ABCMeta, abstractmethod


class BasePolicy(metaclass=ABCMeta):

    @abstractmethod
    def pi(self, x_t, sigma_t):
        """Compute the flow velocity at (x_t, t).

        Args:
            x_t (torch.Tensor): Noisy input at time t.
            sigma_t (torch.Tensor): Noise level at time t.

        Returns:
            torch.Tensor: The computed flow velocity u_t.
        """
        pass

    @abstractmethod
    def detach(self):
        pass
