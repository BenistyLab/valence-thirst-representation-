"""
Neural network architecture for phase-based trial decoding.

Adapted from: Abramson et al. (2026), "Spatial remapping in the subicular
complex and entorhinal cortex follows low-dimensional geometric principles."

Architecture:
  - One shared LSTM temporal encoder (f) that maps calcium activity from any
    phase to a common low-dimensional latent space.
  - Separate feedforward decoder heads (g_c) per phase (pre, airpuff, water)
    that predict trial type (water / airpuff / salt) from the latent state.

    z_T = f(R)   ∈ R^{latent_dim}   (shared encoder, final timestep only)
    ŷ   = g_c(z_T) ∈ R^3            (phase decoder on the latent state)

This mirrors the original paper's LSTM + room-specific MLP decoder design,
but adapted for:
  - Calcium imaging input (not spike counts / firing rates)
  - Three-class trial-type classification output (not x,y regression)
  - Phase-based context (pre / airpuff-SLM / water-SLM) instead of rooms
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared temporal encoder  (f)
# ---------------------------------------------------------------------------

class SharedLSTMEncoder(nn.Module):
    """
    Unidirectional LSTM followed by a small MLP projector.

    Input  shape: (batch, T, n_neurons)
    Output shape: (batch, latent_dim)   <- only the final valid timestep

    Two improvements over the naive implementation:
      1. The MLP projector is applied only to h_T (the final LSTM hidden
         state), not to every timestep.  The intermediate hidden states are
         only needed internally for BPTT; applying the projector to all of
         them was wasteful computation with no gradient benefit.
      2. When `lengths` is provided, h_T is gathered from lstm_out at each
         sequence's true last real timestep (lengths[i] - 1) rather than the
         final padded index.  The LSTM still processes the padding frames but
         we ignore those updates, giving the same h_T as packed sequences
         without the complexity of PackedSequence.

    Paper architecture (Section 4.5):
      LSTM: 512 hidden units
      MLP: 256 -> 128 (ReLU), then linear -> latent_dim (64)

    Current defaults (adapted for this project):
      LSTM: lstm_hidden=32
      MLP:  mlp_hidden=(32,) hidden layer, then linear -> latent_dim=16
      Note: latent_dim controls the MLP *output* (the shared latent space size)
            and must equal the decoder's input dim.  mlp_hidden controls the
            MLP's internal hidden layers and can be set independently.
    """

    def __init__(
        self,
        n_neurons: int,
        lstm_hidden: int = 32,
        mlp_hidden: tuple[int, ...] = (32,),
        latent_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_neurons,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        # MLP projector: maps h_T -> z_T
        layers: list[nn.Module] = []
        in_dim = lstm_hidden
        for h in mlp_hidden:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim))
        self.projector = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (batch, T, n_neurons) - padded calcium activity.
            lengths: (batch,) int64 - actual unpadded length of each trial.
                     When provided, h_T is read from lstm_out at index
                     lengths[i]-1 for each sequence, so zero-padding frames
                     do not affect the extracted hidden state.
                     When None (e.g. called from evaluate without lengths),
                     falls back to the last padded index — correct only when
                     all trials in the batch share the same length.
        Returns:
            z_T: (batch, latent_dim) - latent vector for the final valid timestep.
        """
        lstm_out, _ = self.lstm(x)         # (batch, T, lstm_hidden)

        if lengths is not None:
            # For each sequence pick its true last hidden state.
            # lengths[i] is the number of real frames; index is lengths[i]-1.
            B = lstm_out.size(0)
            idx = (lengths - 1).clamp(min=0).long().to(lstm_out.device)  # (B,)
            h_T = lstm_out[torch.arange(B, device=lstm_out.device), idx]  # (B, lstm_hidden)
        else:
            h_T = lstm_out[:, -1, :]       # (batch, lstm_hidden)

        z_T = self.projector(h_T)          # (batch, latent_dim)
        return z_T


# ---------------------------------------------------------------------------
# Phase-specific decoder head  (g_c)
# ---------------------------------------------------------------------------

class PhaseDecoder(nn.Module):
    """
    Feedforward network that maps a single latent vector z to trial-type logits.

    Architecture (from paper Section 4.5):
      Four hidden layers: 32 -> 32 -> 16 -> 8 (ReLU), then linear -> n_classes.

    Current defaults (adapted for this project):
      Input latent_dim=16 (must match encoder's latent_dim)
      Hidden: (8,) -> then linear -> n_classes=3

    Input:  z  in R^{latent_dim}
    Output: logits in R^{n_classes}
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_sizes: tuple[int, ...] = (8,),
        n_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = latent_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, latent_dim) - latent vector (final timestep of encoder).
        Returns:
            logits: (batch, n_classes)
        """
        return self.net(z)


# ---------------------------------------------------------------------------
# Full model  (encoder + all phase decoders)
# ---------------------------------------------------------------------------

class PhaseDecoderModel(nn.Module):
    """
    Combines the shared LSTM encoder with one decoder head per phase.

    During training each batch is routed to its own phase decoder.
    During cross-phase analysis, a batch from one phase is passed through
    another phase's decoder to reveal how the SLM reshapes representations.

    Parameters
    ----------
    n_neurons:   number of calcium imaging channels (cells).
    phases:      ordered list of phase names, e.g. ['pre', 'airpuff', 'water'].
    lstm_hidden: LSTM hidden-state size (default 32).
    mlp_hidden:  hidden layer sizes inside the encoder's MLP projector
                 (default (32,)).  Controlled independently of latent_dim.
    latent_dim:  output size of the encoder MLP = input size of the decoders.
                 This is the shared latent space dimensionality (default 16).
                 latent_dim and mlp_hidden are intentionally separate so the
                 MLP's internal width can differ from the latent space size.
    dec_hidden:  hidden layer sizes for each phase decoder (default (8,)).
                 The decoder architecture is: latent_dim -> dec_hidden -> n_classes.
    n_classes:   number of trial-type classes (default 3: water, airpuff, salt).
    dropout:     dropout probability applied in both encoder and decoders.
    """

    def __init__(
        self,
        n_neurons: int,
        phases: list[str] = ("pre", "airpuff", "water"),
        lstm_hidden: int = 32,
        mlp_hidden: tuple[int, ...] = (32,),
        latent_dim: int = 16,
        dec_hidden: tuple[int, ...] = (8,),
        n_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.phases = list(phases)
        self.encoder = SharedLSTMEncoder(
            n_neurons=n_neurons,
            lstm_hidden=lstm_hidden,
            mlp_hidden=mlp_hidden,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.decoders = nn.ModuleDict(
            {
                phase: PhaseDecoder(
                    latent_dim=latent_dim,
                    hidden_sizes=dec_hidden,
                    n_classes=n_classes,
                    dropout=dropout,
                )
                for phase in self.phases
            }
        )

    # ------------------------------------------------------------------
    # Core forward: encode + decode with the *own* phase decoder
    # ------------------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return latent vector z_T: (batch, latent_dim).

        Args:
            x:       (batch, T, n_neurons) padded calcium activity.
            lengths: (batch,) int64 actual trial lengths (optional; enables
                     packed-sequence mode so padding does not pollute h_T).
        """
        return self.encoder(x, lengths)

    def decode(self, z: torch.Tensor, phase: str) -> torch.Tensor:
        """
        Decode a latent vector with a named phase decoder.

        Args:
            z:     (batch, latent_dim) - output of encode().
            phase: name of the phase decoder to use.
        Returns:
            logits: (batch, n_classes).
        """
        return self.decoders[phase](z)

    def forward(
        self,
        x: torch.Tensor,
        phase: str,
        lengths: torch.Tensor | None = None,
        decode_phase: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode x and decode with `decode_phase` (defaults to `phase`).

        This is the key method for cross-phase analysis: pass pre-phase
        trials through the SLM decoder by setting decode_phase != phase.

        Args:
            x:            (batch, T, n_neurons) calcium activity.
            phase:        source phase of the input (determines which decoder
                          is used when decode_phase is None).
            lengths:      (batch,) int64 actual trial lengths (optional).
            decode_phase: which decoder to use (defaults to `phase`).
        Returns:
            z_T:    (batch, latent_dim) final latent embedding.
            logits: (batch, n_classes) trial-type logits.
        """
        if decode_phase is None:
            decode_phase = phase
        z_T = self.encode(x, lengths)              # (batch, latent_dim)
        logits = self.decoders[decode_phase](z_T)
        return z_T, logits