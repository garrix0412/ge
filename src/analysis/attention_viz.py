"""
Attention-weight visualisation for Transformer and TFT models.

Extracts variable-selection importance and temporal self-attention patterns
from PyTorch models and renders them as interactive Plotly charts.

Usage::

    from src.analysis.attention_viz import AttentionVisualizer

    viz = AttentionVisualizer(model, feature_names=["close", "volume", ...])
    importance_df = viz.get_feature_importance(X)
    viz.plot_feature_importance(X, save_path="results/figures/feat_imp.html")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn

from src.utils.constants import FIGURES_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AttentionVisualizer:
    """Extract and visualise attention weights from Transformer/TFT models.

    Parameters
    ----------
    model:
        A ``BaseTorchModel`` instance (or anything whose ``.model`` attribute
        is an ``nn.Module``).  The underlying ``nn.Module`` should expose
        attention weights via one of:
        - ``get_attention_weights(x)`` returning a dict
        - A forward hook registered on attention layers
        The class degrades gracefully when these are unavailable.
    feature_names:
        Human-readable names for each input feature, matching the last
        dimension of the model input tensor.
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
    ) -> None:
        # Accept both a wrapper (with .model attribute) and a raw nn.Module
        if hasattr(model, "model") and isinstance(model.model, nn.Module):
            self._module: nn.Module = model.model
            self._device = getattr(model, "device", torch.device("cpu"))
        elif isinstance(model, nn.Module):
            self._module = model
            self._device = torch.device("cpu")
        else:
            raise TypeError(
                f"Expected an nn.Module or a wrapper with a .model attribute, "
                f"got {type(model).__name__}."
            )

        self.feature_names = list(feature_names)
        self._attention_hooks: list[torch.utils.hooks.RemovableHook] = []
        self._captured_attention: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Feature importance (variable selection)
    # ------------------------------------------------------------------

    def get_feature_importance(self, X: np.ndarray) -> pd.DataFrame:
        """Extract variable-selection / feature-importance weights.

        For TFT models this reads the ``variable_selection_weights`` from a
        dedicated forward pass.  For standard Transformers a gradient-based
        proxy (mean absolute gradient w.r.t. input features) is used.

        Parameters
        ----------
        X:
            Input array of shape ``(n_samples, seq_len, n_features)``.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance``.  Sorted descending by
            importance.
        """
        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=self._device)

        # Strategy 1: model exposes get_attention_weights (TFT convention)
        if hasattr(self._module, "get_attention_weights"):
            logger.info("Using model.get_attention_weights() for feature importance.")
            self._module.eval()
            with torch.no_grad():
                attn_dict = self._module.get_attention_weights(x_tensor)
            if "variable_selection" in attn_dict:
                weights = attn_dict["variable_selection"]  # (n_features,) or (n_samples, n_features)
                if isinstance(weights, torch.Tensor):
                    weights = weights.cpu().numpy()
                if weights.ndim > 1:
                    weights = weights.mean(axis=0)
                return self._build_importance_df(weights)

        # Strategy 2: model has a variable_selection_network attribute
        if hasattr(self._module, "variable_selection_network"):
            logger.info("Extracting weights from variable_selection_network.")
            importance = self._extract_vsn_weights()
            if importance is not None:
                return self._build_importance_df(importance)

        # Strategy 3: gradient-based feature importance
        logger.info("Falling back to gradient-based feature importance.")
        return self._gradient_importance(x_tensor)

    def _build_importance_df(self, weights: np.ndarray) -> pd.DataFrame:
        """Build a sorted DataFrame from raw importance weights."""
        weights = np.asarray(weights).ravel()
        n = min(len(weights), len(self.feature_names))
        df = pd.DataFrame(
            {
                "feature": self.feature_names[:n],
                "importance": weights[:n],
            }
        )
        # Normalise to [0, 1]
        total = df["importance"].sum()
        if total > 0:
            df["importance"] = df["importance"] / total
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    def _extract_vsn_weights(self) -> Optional[np.ndarray]:
        """Try to read weights from a Variable Selection Network sub-module."""
        try:
            vsn = self._module.variable_selection_network
            # Typically the first linear layer's weight gives per-feature scores
            for child in vsn.children():
                if isinstance(child, nn.Linear):
                    w = child.weight.detach().cpu().numpy()
                    importance = np.abs(w).mean(axis=0)
                    return importance
        except Exception as exc:
            logger.debug("Could not extract VSN weights: %s", exc)
        return None

    def _gradient_importance(self, x_tensor: torch.Tensor) -> pd.DataFrame:
        """Gradient-based feature importance (mean |grad| w.r.t. input)."""
        self._module.eval()
        x_input = x_tensor.clone().requires_grad_(True)

        try:
            output = self._module(x_input)
            if isinstance(output, tuple):
                output = output[0]
            target = output.sum()
            target.backward()

            grads = x_input.grad.detach().cpu().numpy()  # (n_samples, seq_len, n_features)
            importance = np.abs(grads).mean(axis=(0, 1))  # (n_features,)
        except Exception as exc:
            logger.warning("Gradient importance failed: %s. Returning uniform.", exc)
            n_feat = x_tensor.shape[-1] if x_tensor.ndim >= 2 else len(self.feature_names)
            importance = np.ones(n_feat) / n_feat

        return self._build_importance_df(importance)

    # ------------------------------------------------------------------
    # Temporal attention weights
    # ------------------------------------------------------------------

    def get_temporal_attention(self, X: np.ndarray) -> np.ndarray:
        """Extract temporal self-attention weights from the model.

        Parameters
        ----------
        X:
            Input array of shape ``(n_samples, seq_len, n_features)``.

        Returns
        -------
        np.ndarray
            Attention weights of shape ``(n_samples, n_heads, seq_len, seq_len)``.
            If the model does not expose attention weights, a uniform
            matrix is returned.
        """
        x_tensor = torch.as_tensor(X, dtype=torch.float32, device=self._device)
        n_samples = x_tensor.shape[0]
        seq_len = x_tensor.shape[1] if x_tensor.ndim >= 2 else 1

        self._module.eval()

        # Strategy 1: model exposes get_attention_weights
        if hasattr(self._module, "get_attention_weights"):
            with torch.no_grad():
                attn_dict = self._module.get_attention_weights(x_tensor)
            if "temporal_attention" in attn_dict:
                attn = attn_dict["temporal_attention"]
                if isinstance(attn, torch.Tensor):
                    attn = attn.cpu().numpy()
                logger.info("Temporal attention extracted, shape: %s", attn.shape)
                return attn

        # Strategy 2: register hooks on MultiheadAttention layers
        attn_weights = self._capture_attention_via_hooks(x_tensor)
        if attn_weights is not None:
            logger.info("Temporal attention captured via hooks, shape: %s", attn_weights.shape)
            return attn_weights

        # Fallback: uniform attention
        logger.warning(
            "Could not extract temporal attention — returning uniform weights."
        )
        n_heads = 1
        uniform = np.ones((n_samples, n_heads, seq_len, seq_len)) / seq_len
        return uniform

    def _capture_attention_via_hooks(
        self,
        x_tensor: torch.Tensor,
    ) -> Optional[np.ndarray]:
        """Register forward hooks on MultiheadAttention layers to capture weights."""
        self._captured_attention.clear()

        hooks: list[torch.utils.hooks.RemovableHook] = []

        def _hook_fn(module: nn.Module, inp: Any, out: Any) -> None:
            # nn.MultiheadAttention returns (attn_output, attn_weights)
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                self._captured_attention.append(out[1].detach().cpu())

        # Find all MultiheadAttention layers
        for name, mod in self._module.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                h = mod.register_forward_hook(_hook_fn)
                hooks.append(h)

        if not hooks:
            return None

        try:
            with torch.no_grad():
                _ = self._module(x_tensor)
        except Exception as exc:
            logger.warning("Forward pass for attention capture failed: %s", exc)
            return None
        finally:
            for h in hooks:
                h.remove()

        if not self._captured_attention:
            return None

        # Stack attention from all layers -> take the last layer
        last_attn = self._captured_attention[-1].numpy()
        # Expected shape: (n_samples, seq_len, seq_len) or (n_samples, n_heads, seq_len, seq_len)
        if last_attn.ndim == 3:
            last_attn = last_attn[:, np.newaxis, :, :]  # add head dim
        return last_attn

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def plot_feature_importance(
        self,
        X: np.ndarray,
        save_path: Optional[str | Path] = None,
    ) -> go.Figure:
        """Horizontal bar chart of feature importances.

        Parameters
        ----------
        X:
            Model input array.
        save_path:
            Path to save the figure (HTML).  Defaults to
            ``results/figures/feature_importance.html``.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        importance_df = self.get_feature_importance(X)

        fig = px.bar(
            importance_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Feature Importance (Variable Selection Weights)",
            labels={"importance": "Normalised Importance", "feature": "Feature"},
            color="importance",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            height=max(400, len(importance_df) * 25),
            showlegend=False,
        )

        out_path = Path(save_path) if save_path else FIGURES_DIR / "feature_importance.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        logger.info("Feature importance figure saved -> %s", out_path)

        return fig

    def plot_temporal_attention(
        self,
        X: np.ndarray,
        sample_idx: int = 0,
        save_path: Optional[str | Path] = None,
    ) -> go.Figure:
        """Heatmap of attention weights for a single sample.

        Parameters
        ----------
        X:
            Model input array.
        sample_idx:
            Which sample to visualise.
        save_path:
            Path to save the figure (HTML).

        Returns
        -------
        plotly.graph_objects.Figure
        """
        attn = self.get_temporal_attention(X)
        # attn shape: (n_samples, n_heads, seq_len, seq_len)

        if sample_idx >= attn.shape[0]:
            logger.warning(
                "sample_idx=%d out of range (n_samples=%d). Using 0.",
                sample_idx,
                attn.shape[0],
            )
            sample_idx = 0

        # Average across heads for the selected sample
        attn_sample = attn[sample_idx].mean(axis=0)  # (seq_len, seq_len)

        seq_len = attn_sample.shape[0]
        labels = [f"t-{seq_len - 1 - i}" if i < seq_len - 1 else "t" for i in range(seq_len)]

        fig = go.Figure(
            data=go.Heatmap(
                z=attn_sample,
                x=labels,
                y=labels,
                colorscale="Blues",
                colorbar=dict(title="Weight"),
            )
        )
        fig.update_layout(
            title=f"Temporal Attention Weights (sample {sample_idx}, avg over heads)",
            xaxis_title="Key Position",
            yaxis_title="Query Position",
            height=600,
            width=700,
        )

        out_path = Path(save_path) if save_path else FIGURES_DIR / "temporal_attention.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        logger.info("Temporal attention figure saved -> %s", out_path)

        return fig

    def plot_attention_over_time(
        self,
        X: np.ndarray,
        timestamps: np.ndarray | pd.Series | pd.DatetimeIndex,
        save_path: Optional[str | Path] = None,
    ) -> go.Figure:
        """Visualise how the attention distribution evolves over time.

        For each sample (time step), compute the entropy of the attention
        distribution and a "focus score" (max attention weight).  This
        shows when the model concentrates on specific historical steps
        versus attending broadly.

        Parameters
        ----------
        X:
            Model input array of shape ``(n_samples, seq_len, n_features)``.
        timestamps:
            Timestamps aligned with the sample axis of *X*.
        save_path:
            Path to save the figure (HTML).

        Returns
        -------
        plotly.graph_objects.Figure
        """
        attn = self.get_temporal_attention(X)
        # attn: (n_samples, n_heads, seq_len, seq_len)

        timestamps = pd.to_datetime(timestamps)

        # Average over heads, then for each sample compute summary stats
        # over the last query position (the prediction query)
        attn_avg = attn.mean(axis=1)  # (n_samples, seq_len, seq_len)
        last_query_attn = attn_avg[:, -1, :]  # (n_samples, seq_len) — last query attends to all keys

        # Entropy of the attention distribution
        eps = 1e-12
        entropy = -np.sum(last_query_attn * np.log(last_query_attn + eps), axis=1)

        # Focus score: maximum attention weight
        focus = np.max(last_query_attn, axis=1)

        # Build figure
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Attention Entropy Over Time", "Attention Focus Score Over Time"),
        )

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=entropy,
                mode="lines",
                name="Entropy",
                line=dict(color="steelblue"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=focus,
                mode="lines",
                name="Focus (max weight)",
                line=dict(color="crimson"),
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            height=600,
            title_text="Attention Distribution Over Time",
            showlegend=True,
        )
        fig.update_yaxes(title_text="Entropy", row=1, col=1)
        fig.update_yaxes(title_text="Max Attention", row=2, col=1)

        out_path = Path(save_path) if save_path else FIGURES_DIR / "attention_over_time.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out_path))
        logger.info("Attention-over-time figure saved -> %s", out_path)

        return fig
