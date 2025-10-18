import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    """Remove padding from the right side of the input"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """Temporal block with causal dilated convolution and residual connection"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Combine operations
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize weights for better training stability"""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """Forward pass with residual connection"""
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class OptimizedTCN(nn.Module):
    """Optimized TCN with 2 layers for reduced complexity"""
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout=0.2):
        super(OptimizedTCN, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Use kernel_size=5 for the last layer (dilation=4)
            current_kernel_size = kernel_size if dilation_size != 4 else 5
            
            layers += [TemporalBlock(
                in_channels, out_channels, current_kernel_size, 
                stride=1, dilation=dilation_size,
                padding=(current_kernel_size-1) * dilation_size, 
                dropout=dropout
            )]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            output: Output tensor of shape (batch_size, sequence_length, output_size)
        """
        # Convert to (batch_size, input_size, sequence_length) for Conv1d
        x = x.permute(0, 2, 1)
        
        # Apply TCN layers
        output = self.network(x)
        
        # Convert back to (batch_size, sequence_length, output_size)
        output = output.permute(0, 2, 1)
        
        # Apply final linear layer
        output = self.linear(output)
        
        return output

class OptimizedMultiScaleTCN(nn.Module):
    """Optimized multi-scale TCN with reduced hidden dimensions"""
    def __init__(self, input_dim: int, sequence_length: int, hidden_dim: int = 64, dropout: float = 0.2):
        super(OptimizedMultiScaleTCN, self).__init__()
        
        # Three TCN branches with optimized dimensions
        self.tcn_branch1 = OptimizedTCN(
            input_dim, hidden_dim, [hidden_dim] * 2, 
            kernel_size=3, dropout=dropout
        )
        self.tcn_branch2 = OptimizedTCN(
            input_dim, hidden_dim // 2, [hidden_dim // 2] * 2, 
            kernel_size=3, dropout=dropout
        )
        self.tcn_branch3 = OptimizedTCN(
            input_dim, hidden_dim * 2, [hidden_dim * 2] * 2, 
            kernel_size=3, dropout=dropout
        )
        
        # Calculate total output dimension
        self.output_dim = hidden_dim + (hidden_dim // 2) + (hidden_dim * 2)

    def forward(self, x):
        """
        Forward pass through multi-scale TCN branches
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
        Returns:
            combined_features: Pooled features of shape (batch_size, total_dim)
        """
        # Process through each TCN branch
        out1 = self.tcn_branch1(x)  # (batch_size, sequence_length, hidden_dim)
        out2 = self.tcn_branch2(x)  # (batch_size, sequence_length, hidden_dim // 2)
        out3 = self.tcn_branch3(x)  # (batch_size, sequence_length, hidden_dim * 2)
        
        # Pool the last time step from each branch
        pooled_out1 = out1[:, -1, :]  # (batch_size, hidden_dim)
        pooled_out2 = out2[:, -1, :]  # (batch_size, hidden_dim // 2)
        pooled_out3 = out3[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # Concatenate pooled outputs
        combined_features = torch.cat([pooled_out1, pooled_out2, pooled_out3], dim=1)
        
        return combined_features
