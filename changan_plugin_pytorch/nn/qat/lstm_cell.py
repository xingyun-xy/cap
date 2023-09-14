import torch
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.utils.model_helper import assert_qtensor_input

from ...dtype import _storage_type_map
from .linear import Linear
from .segment_lut import SegmentLUT


class LSTMCell(torch.nn.Module):
    _FLOAT_MODULE = torch.nn.LSTMCell

    def __init__(self, input_size, hidden_size, qconfig):
        super(LSTMCell, self).__init__()
        from changan_plugin_pytorch.nn.quantized.functional_modules import (
            FloatFunctional,
        )

        self.qconfig = qconfig
        self.out_type = qconfig.activation().dtype
        self.hidden_size = hidden_size
        self.input_linear = Linear(
            in_features=input_size,
            out_features=4 * hidden_size,
            bias=False,
            qconfig=qconfig,
        )
        self.hidden_linear = Linear(
            in_features=hidden_size,
            out_features=4 * hidden_size,
            bias=False,
            qconfig=qconfig,
        )
        self.i2h_h2h_add = FloatFunctional(qconfig=qconfig)
        self.fg_cur_state_mul = FloatFunctional(qconfig=qconfig)
        self.ig_it_mul = FloatFunctional(qconfig=qconfig)
        self.next_state_add = FloatFunctional(qconfig=qconfig)
        self.next_hid_state_mul = FloatFunctional(qconfig=qconfig)
        self.in_gate_sigmoid = SegmentLUT(
            torch.sigmoid,
            False,
            None,
            qconfig=qconfig,
        )
        self.forget_gate_sigmoid = SegmentLUT(
            torch.sigmoid,
            False,
            None,
            qconfig=qconfig,
        )
        self.in_transform_tanh = SegmentLUT(
            torch.tanh,
            False,
            None,
            qconfig=qconfig,
        )
        self.out_gate_sigmoid = SegmentLUT(
            torch.sigmoid,
            False,
            None,
            qconfig=qconfig,
        )
        self.next_state_tanh = SegmentLUT(
            torch.tanh,
            False,
            None,
            qconfig=qconfig,
        )

    def forward(self, data, states=None):
        assert_qtensor_input(data)

        # TODO: torch1.11 support one dim input
        if states is None:
            device = data.as_subclass(torch.Tensor).device
            if isinstance(self.input_linear, Linear):
                if not hasattr(self, "hid_state"):
                    st_shape = (data.shape[0], self.hidden_size)
                    self.hid_state = torch.zeros(st_shape, device=device)
                    self.cur_state = torch.zeros(st_shape, device=device)
                hid_state = QTensor(
                    self.hid_state, data.q_scale(), self.out_type
                )
                cur_state = QTensor(
                    self.cur_state, data.q_scale(), self.out_type
                )
            else:
                hid_state = QTensor(
                    self.hid_state.to(_storage_type_map[self.out_type]),
                    data.q_scale(),
                    self.out_type,
                )
                cur_state = QTensor(
                    self.cur_state.to(_storage_type_map[self.out_type]),
                    data.q_scale(),
                    self.out_type,
                )
        else:
            hid_state, cur_state = states
        input_to_hidden = self.input_linear(data)
        hidden_to_hidden = self.hidden_linear(hid_state)
        split_dim = 1
        if not isinstance(self.input_linear, Linear):
            i2h_shape = input_to_hidden.shape
            h2h_shape = hidden_to_hidden.shape
            data_shape = data.shape
            cs_shape = cur_state.shape
            input_to_hidden = input_to_hidden.reshape(
                1, i2h_shape[0], 1, i2h_shape[1]
            )
            hidden_to_hidden = hidden_to_hidden.reshape(
                1, h2h_shape[0], 1, h2h_shape[1]
            )
            data = data.reshape(1, data_shape[0], 1, data_shape[1])
            cur_state = cur_state.reshape(1, cs_shape[0], 1, cs_shape[1])
            split_dim = 3
        gate = self.i2h_h2h_add.add(input_to_hidden, hidden_to_hidden)
        slice_gates = torch.split(
            gate, split_size_or_sections=self.hidden_size, dim=split_dim
        )
        in_gate = self.in_gate_sigmoid(slice_gates[0])
        forget_gate = self.forget_gate_sigmoid(slice_gates[1])
        in_transform = self.in_transform_tanh(slice_gates[2])
        out_gate = self.out_gate_sigmoid(slice_gates[3])
        next_state = self.next_state_add.add(
            self.fg_cur_state_mul.mul(forget_gate, cur_state),
            self.ig_it_mul.mul(in_gate, in_transform),
        )
        next_hidden_state = self.next_hid_state_mul.mul(
            out_gate, self.next_state_tanh(next_state)
        )
        if not isinstance(self.input_linear, Linear):
            next_hidden_state = next_hidden_state.reshape(hid_state.shape)
            next_state = next_state.reshape(cs_shape)
        return next_hidden_state, next_state

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

        Args: `mod` a float module
        """
        assert type(mod) == cls._FLOAT_MODULE, (
            "qat."
            + cls.__name__
            + ".from_float only works for "
            + cls._FLOAT_MODULE.__name__
        )
        assert mod.qconfig, "Input float module must have a valid qconfig"
        qconfig = mod.qconfig
        qat_lstm_cell = cls(
            input_size=mod.input_size,
            hidden_size=mod.hidden_size,
            qconfig=qconfig,
        )
        qat_lstm_cell.input_linear.weight = mod.weight_ih
        qat_lstm_cell.input_linear.bias = mod.bias_ih
        qat_lstm_cell.hidden_linear.weight = mod.weight_hh
        qat_lstm_cell.hidden_linear.bias = mod.bias_hh
        return qat_lstm_cell
