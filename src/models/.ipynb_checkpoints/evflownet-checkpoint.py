import torch
from torch import nn
from typing import Dict, Any
from src.models.base import build_resnet_block, upsample_conv2d_and_predict_flow, general_conv2d

_BASE_CHANNELS = 64

class EVFlowNet(nn.Module):
    def __init__(self, args):
        super(EVFlowNet, self).__init__()
        self._args = args

        self.encoder1 = general_conv2d(in_channels=8, out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)  # ここを4から8に変更
        self.encoder2 = general_conv2d(in_channels=_BASE_CHANNELS, out_channels=2 * _BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder3 = general_conv2d(in_channels=2 * _BASE_CHANNELS, out_channels=4 * _BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder4 = general_conv2d(in_channels=4 * _BASE_CHANNELS, out_channels=8 * _BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.resnet_block = nn.Sequential(
            *[build_resnet_block(8 * _BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm) for _ in range(2)]
        )

        self.decoder1 = upsample_conv2d_and_predict_flow(in_channels=16 * _BASE_CHANNELS,
                                                         out_channels=4 * _BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.decoder2 = upsample_conv2d_and_predict_flow(in_channels=8 * _BASE_CHANNELS + 2,
                                                         out_channels=2 * _BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.decoder3 = upsample_conv2d_and_predict_flow(in_channels=4 * _BASE_CHANNELS + 2,
                                                         out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.decoder4 = upsample_conv2d_and_predict_flow(in_channels=2 * _BASE_CHANNELS + 2,
                                                         out_channels=int(_BASE_CHANNELS / 2), do_batch_norm=not self._args.no_batch_norm)

        self.criterion = nn.MSELoss()  # Example loss function, can be changed as per your requirement

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Encoder
        skip_connections = {}
        inputs = self.encoder1(inputs)
        skip_connections['skip0'] = inputs.clone()
        inputs = self.encoder2(inputs)
        skip_connections['skip1'] = inputs.clone()
        inputs = self.encoder3(inputs)
        skip_connections['skip2'] = inputs.clone()
        inputs = self.encoder4(inputs)
        skip_connections['skip3'] = inputs.clone()

        # Transition
        inputs = self.resnet_block(inputs)

        # Decoder and intermediate loss computation
        flow_dict = {}
        loss = 0.0

        inputs = torch.cat([inputs, skip_connections['skip3']], dim=1)
        inputs, flow = self.decoder1(inputs)
        flow_dict['flow0'] = flow.clone()
        loss += self.compute_intermediate_loss(flow, skip_connections['skip3'], inputs.device)

        inputs = torch.cat([inputs, skip_connections['skip2']], dim=1)
        inputs, flow = self.decoder2(inputs)
        flow_dict['flow1'] = flow.clone()
        loss += self.compute_intermediate_loss(flow, skip_connections['skip2'], inputs.device)

        inputs = torch.cat([inputs, skip_connections['skip1']], dim=1)
        inputs, flow = self.decoder3(inputs)
        flow_dict['flow2'] = flow.clone()
        loss += self.compute_intermediate_loss(flow, skip_connections['skip1'], inputs.device)

        inputs = torch.cat([inputs, skip_connections['skip0']], dim=1)
        inputs, flow = self.decoder4(inputs)
        flow_dict['flow3'] = flow.clone()
        loss += self.compute_intermediate_loss(flow, skip_connections['skip0'], inputs.device)

        return flow_dict, loss

    def compute_intermediate_loss(self, flow, skip_connection, device):
        downsampled_skip = nn.functional.interpolate(skip_connection, size=flow.shape[2:], mode='bilinear', align_corners=True)
        reduced_skip = self.reduce_channels(downsampled_skip, flow.shape[1], device)
        return self.criterion(flow, reduced_skip)

    def reduce_channels(self, tensor, target_channels, device):
        conv = nn.Conv2d(tensor.shape[1], target_channels, kernel_size=1).to(device)
        return conv(tensor)
