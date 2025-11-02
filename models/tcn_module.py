"""
Temporal Convolutional Network (TCN) Module for Multi-Scale Feature Extraction

This module implements a TCN with causal dilated convolutions for capturing
short-, medium-, and long-term temporal patterns in network traffic sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TCNBlock(nn.Module):
    """Single TCN block with causal dilated convolution and residual connection."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int, dropout: float = 0.2):
        super(TCNBlock, self).__init__()
        
        # Calculate padding for causal convolution
        # For causal conv: padding = (kernel_size - 1) * dilation
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Residual connection (1x1 conv if dimensions don't match)
        self.residual_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        ) if in_channels != out_channels else None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN block.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, sequence_length)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, sequence_length)
        """
        residual = x
        
        # First convolution
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolution
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Apply residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        
        # Ensure causal behavior by truncating future information
        # Remove the last (kernel_size - 1) * dilation elements
        if out.size(-1) > residual.size(-1):
            out = out[:, :, :residual.size(-1)]
        
        out = out + residual
        return out


class TCN(nn.Module):
    """
    Temporal Convolutional Network for multi-scale temporal pattern extraction.
    
    This TCN uses three layers with different dilations to capture:
    - Short-term patterns (dilation=1)
    - Medium-term patterns (dilation=2) 
    - Long-term patterns (dilation=4)
    """
    
    def __init__(self, input_dim: int = 40, hidden_dim: int = 128, 
                 num_layers: int = 3, kernel_size: int = 3, 
                 dropout: float = 0.2, sequence_length: int = 30):
        super(TCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # Define dilations for multi-scale feature extraction
        # dilation=1: short-term patterns (kernel_size=3)
        # dilation=2: medium-term patterns (kernel_size=3)
        # dilation=4: long-term patterns (kernel_size=5)
        self.dilations = [1, 2, 4]
        self.kernel_sizes = [3, 3, 5]  # Use larger kernel for long-term patterns
        
        # Input projection
        self.input_projection = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=1
        )
        
        # TCN blocks
        self.tcn_blocks = nn.ModuleList()
        
        in_channels = hidden_dim
        for i, (dilation, kernel_size) in enumerate(zip(self.dilations, self.kernel_sizes)):
            self.tcn_blocks.append(
                TCNBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
            in_channels = hidden_dim
        
        # Final normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through TCN.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, sequence_length, hidden_dim)
        """
        # Input: (batch_size, sequence_length, input_dim)
        # Transpose for Conv1d: (batch_size, input_dim, sequence_length)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_projection(x)
        
        # Apply TCN blocks
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)
        
        # Transpose back: (batch_size, sequence_length, hidden_dim)
        x = x.transpose(1, 2)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        return x
    
    def get_output_dim(self) -> int:
        """Get the output dimension of the TCN."""
        return self.hidden_dim


class MultiScaleTCN(nn.Module):
    """
    Multi-scale TCN with different hidden dimensions for capturing
    various temporal scales and complexities.
    """
    
    def __init__(self, input_dim: int = 40, hidden_dims: list = [128, 64, 256],
                 sequence_length: int = 30, dropout: float = 0.2):
        super(MultiScaleTCN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.sequence_length = sequence_length
        self.total_output_dim = sum(hidden_dims)
        
        # Create multiple TCN branches with different hidden dimensions
        self.tcn_branches = nn.ModuleList()
        
        for hidden_dim in hidden_dims:
            self.tcn_branches.append(
                TCN(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    sequence_length=sequence_length,
                    dropout=dropout
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale TCN.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Concatenated output tensor of shape (batch_size, total_output_dim)
        """
        batch_size = x.size(0)
        outputs = []
        
        # Process through each TCN branch
        for tcn_branch in self.tcn_branches:
            # Get output: (batch_size, sequence_length, hidden_dim)
            branch_output = tcn_branch(x)
            
            # Pool last time step: (batch_size, hidden_dim)
            last_timestep = branch_output[:, -1, :]
            outputs.append(last_timestep)
        
        # Concatenate all branch outputs
        # Result: (batch_size, total_output_dim)
        concatenated = torch.cat(outputs, dim=1)
        
        return concatenated
    
    def get_output_dim(self) -> int:
        """Get the total output dimension of the multi-scale TCN."""
        return self.total_output_dim





