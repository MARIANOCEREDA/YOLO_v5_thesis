import torch
import torch.nn as nn
from config.config import architecture_config

class CNNBlock(nn.Module):

    def __init__(self, in_channels:int, out_channels:int, bn_act:bool=True, **kwargs) -> None:

        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)

        self.bn = nn.BatchNorm2d(out_channels)

        self.leaky = nn.LeakyReLU(0.1)

        self.use_bn_act = bn_act
    

    def forward(self, x:torch.Tensor):
        """
        Performs the forward pass of the network

        Parameters:
            x (Tensor) : Input data
        
        Returns:
            The output of the network.
        """

        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):

    def __init__(self, channels:int, use_residual:bool=True, num_repeats:int=1) -> None:

        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        
        self.num_repeats = num_repeats

    def forward(self, x:torch.Tensor):
        """
        Performs the forward pass of the network

        Parameters:
            x (Tensor) : Input data
        
        Returns:
            The output of the network. It can be:
                - A residual block: Add the input to the output of the conv layer
                - The output of the conv layer itself.
        """

        for conv_layer in self.layers:

            if self.use_residual:
                x = x + conv_layer(x)

            else:
                x = conv_layer(x)

        return x

class ScalePrediction(nn.Module):

    def __init__(self, in_channels:int, num_classes:int = 1):

        super().__init__()

        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1),
        )

        self.num_classes = num_classes

    def forward(self, x:torch.Tensor):
        """
        Performs the forward pass of the network

        Parameters:
            x (Tensor) : Input data
        
        Returns:
            The output of the network. 

            For example:
                - For the first prediction scale (13 x 13): N (batch-size) x 3(num_anchors) x 13 x 13 x (5 + num_classes)
                - For the second prediction scale (26 x 26): N (batch-size) x 3(num_anchors) x 26 x 26 x (5 + num_classes)
                - For the first prediction scale: x.shape = (N, 13, 13)
        """

        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

class YOLOv3(nn.Module):

    def __init__(self, in_channels:int, arch_config:list, num_classes:int = 1) -> None:

        super().__init__()

        self.in_channels = in_channels

        self.config = arch_config
        
        self.num_classes = num_classes

        self.conv_layers = self._create_conv_layers()
    
    def forward(self, x:torch.Tensor) -> list:

        outputs = []  

        route_connections = []

        for layer in self.conv_layers:

            if isinstance(layer, ScalePrediction):

                outputs.append(layer(x))

                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:

                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):

                x = torch.cat([x, route_connections[-1]], dim=1)

                route_connections.pop()

        return outputs


    def _create_conv_layers(self) -> nn.ModuleList:

        layers = nn.ModuleList()

        in_channels = self.in_channels

        for module in self.config:

            if module["type"] == "Conv":

                out_channels, kernel_size, stride = module["size"]

                layers.append(
                    CNNBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size==3 else 0
                    )
                )

                in_channels = out_channels
            
            elif module["type"] == "Residual":

                num_repeats = module["num_repeats"]

                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))
            
            elif module["type"] == "Detect":
                
                layers += [
                    ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                    CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                    ScalePrediction(in_channels // 2, num_classes=self.num_classes)
                ]

                in_channels = in_channels // 2
            
            elif module["type"] == "Upsample":

                layers.append(nn.Upsample(scale_factor=2))

                in_channels = in_channels * 3
        
        return layers

if __name__ == "__main__":
    num_classes = 1
    IMAGE_SIZE = 416
    BATCH_SIZE = 1
    arch = architecture_config()


    model = YOLOv3(in_channels=3, num_classes=num_classes, arch_config=arch)

    x = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE))

    print(f"Input shape: {x.shape}")

    out = model(x)

    print(out[0].shape)

    assert out[0].shape == (BATCH_SIZE, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert out[1].shape == (BATCH_SIZE, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert out[2].shape == (BATCH_SIZE, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")



 