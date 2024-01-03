from torch import nn

## Your solution here
class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = self.linear_layers(784, 64)
        self.output = self.final_layer(64, 10)
        
    def linear_layers(self, input_dim, output_dim):
        x = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Dropout(p = 0.37),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

        return x

    def final_layer(self, input_dim, output_dim):
        x = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Dropout(p = 0.37),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
            # nn.Softmax()
        )

        return x
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.fc1(x)
        x = self.output(x)
        return x
    
