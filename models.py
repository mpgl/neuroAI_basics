import torch
import torch.nn as nn


class CUBAPointLeaky(nn.Module):
    """
    A Current-Based Leaky Integrate-and-Fire (CUBA-LIF) neuron.
    Spikes induce a decaying synaptic current, which then drives the membrane potential.
    """
    def __init__(
        self,
        threshold: float,
        reset: float,
        tau_mem: float,
        tau_syn: float,
        R: float,
        dt: float,
    ):
        super().__init__()
        self.threshold = threshold
        self.reset = reset

        # Precompute decay factors (alphas) using the exponential form
        self.alpha_mem = torch.exp(torch.tensor(-dt / tau_mem)) # membrane voltage
        self.alpha_syn = torch.exp(torch.tensor(-dt / tau_syn)) # synaptic current

        # Pre-calculate the voltage scaling term: R * (1 - alpha_mem)
        # This converts the total Current (Amperes) into a Voltage step.
        self.voltage_scale = R * (1.0 - self.alpha_mem)

    def initialize_state(self, batch_size: int, num_neurons: int, device: torch.device):
        """Initialize the state tuple (Membrane, Synaptic Current)"""
        shape = (batch_size, num_neurons)
        mem = torch.zeros(shape, device=device)
        isyn = torch.zeros(shape, device=device)
        return mem, isyn

    def forward(
        self,
        x,
        state: tuple[torch.Tensor, torch.Tensor],
        use_surrogate: bool = False,
        surrogate_alpha: float = 2.0
    ):
        """
        x: Incoming spikes (delta inputs)
        state: Tuple (mem, isyn)
        """
        mem, isyn = state

        # Update synaptic current
        new_isyn = self.alpha_syn * isyn + x

        # Update membrane potential
        new_mem = self.alpha_mem * mem + self.voltage_scale * new_isyn

        # Generate spikes using either surrogate gradient or hard threshold
        if use_surrogate:
            spike = ATanSurrogate.apply(new_mem - self.threshold, surrogate_alpha)
        else:
            spike = (new_mem > self.threshold).float()

        # Reset membrane potential where spikes occurred
        mem = new_mem * (1 - spike.detach()) + self.reset * spike.detach() # type: ignore

        return spike, (mem, new_isyn)


class ATanSurrogate(torch.autograd.Function):
    """
    A surrogate gradient function using the derivative of the arctangent. This
    custom autograd function implements the Heaviside step function in the
    forward pass and its surrogate gradient (based on the arctangent function's
    derivative) in the backward pass. The steepness of the gradient is
    controlled by the `alpha` parameter.
    """
    @staticmethod
    def forward(ctx, x, alpha=2.0):
        """
        Applies the Heaviside step function in the forward pass.

        Args:
            ctx: A context object to save information for the backward pass.
            x (torch.Tensor): The input tensor (e.g., membrane potential minus threshold).
            alpha (float, optional): The steepness parameter for the gradient. Defaults to 2.0.

        Returns:
            torch.Tensor: A binary tensor, where elements are 1.0 if the corresponding
                          input element is greater than 0, and 0.0 otherwise.
        """
        ctx.save_for_backward(x)
        ctx.alpha = alpha  # Save for use in the backward pass
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradient using the arctangent derivative surrogate.

        Args:
            ctx: The context object with saved tensors from the forward pass.
            grad_output (torch.Tensor): The gradient of the loss with respect to the
                                        output of the forward pass.

        Returns:
            Tuple[torch.Tensor, None]: A tuple containing the gradient with respect to the
                                       input `x`, and None for the `alpha` parameter as it
                                       does not require a gradient.
        """
        x, = ctx.saved_tensors
        alpha = ctx.alpha
        grad_x = (alpha / 2) / (1.0 + (torch.pi / 2 * alpha * x).pow(2))
        return grad_output * grad_x, None


class CUBAPointSurrogateNet(nn.Module):
    def __init__(
            self,
            num_inputs: int,
            num_hidden: int,
            num_outputs: int,
            num_steps: int,
            threshold: float,
            reset: float,
            tau_mem: float,
            tau_syn: float,
            R: float,
            dt: float,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.num_steps = num_steps

        # ---- Initialize layers ----------------------------------------------#
        self.weights_hidden = nn.Linear(num_inputs, num_hidden, bias=False)
        self.lif_hidden = CUBAPointLeaky(
            threshold=threshold,
            reset=reset,
            R=R,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            dt=dt)

        self.weights_out = nn.Linear(num_hidden, num_outputs, bias=False)
        self.lif_out = CUBAPointLeaky(
            threshold=threshold,
            reset=reset,
            R=R,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            dt=dt)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1) # Flatten input
        B = x.size(0) # Batch size

        # Initialize membrane values at t=0
        mem_hidden, isyn_hidden = self.lif_hidden.initialize_state(
            batch_size=B,
            num_neurons=self.num_hidden,
            device=x.device)
        mem_out, isyn_out = self.lif_out.initialize_state(
            batch_size=B,
            num_neurons=self.num_outputs,
            device=x.device)

        # Create empty tensor to count output spikes
        spk_count_output = torch.zeros(B, self.num_outputs, device=x.device)

        # Iterate through each timestep
        for step in range(self.num_steps):
            # Generate input spikes for this timestep
            spk_input = torch.bernoulli(x)

            # Forward pass through the SNN
            cur_hidden = self.weights_hidden(spk_input)
            spk_hidden, (mem_hidden, isyn_hidden) = self.lif_hidden(
                cur_hidden, (mem_hidden, isyn_hidden), use_surrogate=True)
            cur_out = self.weights_out(spk_hidden)
            spk_out, (mem_out, isyn_out) = self.lif_out(
                cur_out, (mem_out, isyn_out), use_surrogate=True)

            # Accumulate output spikes
            spk_count_output += spk_out

        return spk_count_output
