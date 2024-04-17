from torch import nn


class tinyVGGModel(nn.Module):
    def __init__(self, in_shape, hidden_units, out_shape):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU()
        )
        self.second_conv = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.third_conv = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=13 * 13 * hidden_units,
                      out_features=out_shape)
        )

    def forward(self, x):
        return self.classifier((self.third_conv(self.second_conv(self.first_conv(x)))))
