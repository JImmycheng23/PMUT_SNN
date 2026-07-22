import torch
from torch import nn


class SpikeFunction(torch.autograd.Function):
    """Heaviside spike with fast-sigmoid surrogate gradient."""

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.slope = slope
        return (input_tensor > 0.0).to(input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (input_tensor,) = ctx.saved_tensors
        slope = ctx.slope
        grad = grad_output / (slope * torch.abs(input_tensor) + 1.0) ** 2
        return grad, None


def spike(input_tensor: torch.Tensor, slope: float = 25.0) -> torch.Tensor:
    return SpikeFunction.apply(input_tensor, slope)


class LIFStateLayer(nn.Module):
    def __init__(self, beta: float = 0.9, threshold: float = 1.0, surrogate_slope: float = 25.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.surrogate_slope = surrogate_slope

    def forward(self, input_current: torch.Tensor, membrane: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if membrane is None:
            membrane = torch.zeros_like(input_current)
        membrane = self.beta * membrane + input_current
        spikes = spike(membrane - self.threshold, self.surrogate_slope)
        membrane = membrane - spikes.detach() * self.threshold
        return spikes, membrane


class SpikeActivationHook:
    def __init__(self, name: str):
        self.name = name
        self.activations = []

    def __call__(self, module, input_tensor, output):
        self.activations.append(output.detach().cpu())

    def get_activation_map(self, clear: bool = True):
        if not self.activations:
            return None
        result = torch.cat(self.activations, dim=0)
        if clear:
            self.activations.clear()
        return result


class RateSNNMLP(nn.Module):
    """Rate-coded static MLP with LIF hidden layers with BatchNorm."""

    def __init__(self, input_shape: tuple[int, int], num_classes: int, hidden_size: int = 128, steps: int = 16):
        super().__init__()
        seq_len, channels = input_shape
        in_features = seq_len * channels
        self.steps = steps
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        self.lif1 = LIFStateLayer(beta=0.9)
        self.hidden = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        self.lif2 = LIFStateLayer(beta=0.9)
        self.readout = nn.Linear(hidden_size, num_classes)
        self._hooks: list[SpikeActivationHook] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        static_current = self.encoder(x)
        mem1 = None
        mem2 = None
        logits_sum = 0.0
        for _ in range(self.steps):
            spk1, mem1 = self.lif1(static_current, mem1)
            spk2, mem2 = self.lif2(self.hidden(spk1), mem2)
            logits_sum = logits_sum + self.readout(spk2)
        return logits_sum / self.steps

    def register_spike_hooks(self) -> list[SpikeActivationHook]:
        hook1 = SpikeActivationHook("lif1")
        hook2 = SpikeActivationHook("lif2")
        self.lif1.register_forward_hook(lambda m, i, o: hook1(m, i, o[0]))
        self.lif2.register_forward_hook(lambda m, i, o: hook2(m, i, o[0]))
        self._hooks = [hook1, hook2]
        return self._hooks

    def get_spike_rates(self) -> dict[str, float]:
        result = {}
        for hook in self._hooks:
            maps = hook.get_activation_map(clear=False)
            if maps is not None:
                result[hook.name] = float(maps.float().mean().item())
        return result


class TemporalLIFSNN(nn.Module):
    """LIF recurrent-style SNN that consumes frequency sweep points as time, with BatchNorm."""

    def __init__(self, input_shape: tuple[int, int], num_classes: int, hidden_size: int = 128):
        super().__init__()
        _, channels = input_shape
        self.input_proj = nn.Sequential(
            nn.Linear(channels, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        self.lif = LIFStateLayer(beta=0.92)
        self.readout = nn.Linear(hidden_size, num_classes)
        self._hooks: list[SpikeActivationHook] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        membrane = None
        logits_sum = 0.0
        for t in range(x.shape[1]):
            current = self.input_proj(x[:, t, :])
            spikes, membrane = self.lif(current, membrane)
            logits_sum = logits_sum + self.readout(spikes)
        return logits_sum / x.shape[1]

    def register_spike_hooks(self) -> list[SpikeActivationHook]:
        hook = SpikeActivationHook("lif")
        self.lif.register_forward_hook(lambda m, i, o: hook(m, i, o[0]))
        self._hooks = [hook]
        return self._hooks

    def get_spike_rates(self) -> dict[str, float]:
        result = {}
        for hook in self._hooks:
            maps = hook.get_activation_map(clear=False)
            if maps is not None:
                result[hook.name] = float(maps.float().mean().item())
        return result


class SpikingCNN1D(nn.Module):
    """1D convolution over sweep axis followed by LIF spiking blocks with BatchNorm."""

    def __init__(self, input_shape: tuple[int, int], num_classes: int, hidden_channels: int = 32):
        super().__init__()
        _, channels = input_shape
        self.conv1 = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels * 2),
        )
        self.pool = nn.AvgPool1d(kernel_size=2)
        self.lif1 = LIFStateLayer(beta=0.9)
        self.lif2 = LIFStateLayer(beta=0.9)
        self.head = nn.Linear(hidden_channels * 2, num_classes)
        self._hooks: list[SpikeActivationHook] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        z1 = self.conv1(x)
        spk1, _ = self.lif1(z1)
        z2 = self.pool(spk1)
        z2 = self.conv2(z2)
        spk2, _ = self.lif2(z2)
        pooled = spk2.mean(dim=-1)
        return self.head(pooled)

    def register_spike_hooks(self) -> list[SpikeActivationHook]:
        hook1 = SpikeActivationHook("lif1")
        hook2 = SpikeActivationHook("lif2")
        self.lif1.register_forward_hook(lambda m, i, o: hook1(m, i, o[0]))
        self.lif2.register_forward_hook(lambda m, i, o: hook2(m, i, o[0]))
        self._hooks = [hook1, hook2]
        return self._hooks

    def get_spike_rates(self) -> dict[str, float]:
        result = {}
        for hook in self._hooks:
            maps = hook.get_activation_map(clear=False)
            if maps is not None:
                result[hook.name] = float(maps.float().mean().item())
        return result


class CNN1D(nn.Module):
    """Non-spiking 1D CNN baseline for hybrid CNN/SNN comparison."""

    def __init__(self, input_shape: tuple[int, int], num_classes: int, hidden_channels: int = 32):
        super().__init__()
        _, channels = input_shape
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=9, padding=4),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        features = self.net(x).squeeze(-1)
        return self.head(features)


class EncodedSpikeSNN(nn.Module):
    """SNN that processes pre-encoded spike trains from NB8 encoders.

    This is the model for the intended pipeline:
        PMUT sweep → explicit encoder (rate/latency/delta/etc.) → this SNN → class

    Input: (batch, n_steps, n_features) — binary spike trains.
    n_features depends on the encoder:
        rate/latency: window_size   threshold: n_thresholds (6)
        delta: 2                    population: n_neurons (16)
    """

    def __init__(self, n_features: int, num_classes: int, hidden_size: int = 128):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        self.lif = LIFStateLayer(beta=0.9)
        self.readout = nn.Linear(hidden_size, num_classes)
        self._hooks: list[SpikeActivationHook] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        membrane = None
        logits_sum = 0.0
        for t in range(x.shape[1]):
            current = self.input_proj(x[:, t, :])
            spikes, membrane = self.lif(current, membrane)
            logits_sum = logits_sum + self.readout(spikes)
        return logits_sum / x.shape[1]

    def register_spike_hooks(self) -> list[SpikeActivationHook]:
        hook = SpikeActivationHook("lif")
        self.lif.register_forward_hook(lambda m, i, o: hook(m, i, o[0]))
        self._hooks = [hook]
        return self._hooks

    def get_spike_rates(self) -> dict[str, float]:
        result = {}
        for hook in self._hooks:
            maps = hook.get_activation_map(clear=False)
            if maps is not None:
                result[hook.name] = float(maps.float().mean().item())
        return result


def build_model(
    model_name: str,
    input_shape: tuple[int, int],
    num_classes: int,
    n_features: int | None = None,
) -> nn.Module:
    if model_name == "rate_snn_mlp":
        return RateSNNMLP(input_shape=input_shape, num_classes=num_classes)
    if model_name == "temporal_lif_snn":
        return TemporalLIFSNN(input_shape=input_shape, num_classes=num_classes)
    if model_name == "spiking_cnn1d":
        return SpikingCNN1D(input_shape=input_shape, num_classes=num_classes)
    if model_name == "cnn1d":
        return CNN1D(input_shape=input_shape, num_classes=num_classes)
    if model_name == "encoded_spike_snn":
        if n_features is None:
            raise ValueError("encoded_spike_snn requires n_features (encoder output dim)")
        return EncodedSpikeSNN(n_features=n_features, num_classes=num_classes)
    raise ValueError(f"Unknown model: {model_name}")

