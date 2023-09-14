import torch
from changan_plugin_pytorch.dtype import _storage_type_map
from changan_plugin_pytorch.qtensor import QTensor
from changan_plugin_pytorch.utils.model_helper import assert_qtensor_input

from .dropout import Dropout
from .linear import Linear
from .segment_lut import SegmentLUT


class LSTM(torch.nn.Module):
    _FLOAT_MODULE = torch.nn.LSTM

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias,
        batch_first,
        dropout,
        bidirectional,
        proj_size,
        qconfig,
    ):
        super(LSTM, self).__init__()
        from changan_plugin_pytorch.nn.quantized import FloatFunctional

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.qconfig = qconfig
        self.out_type = qconfig.activation().dtype
        if bidirectional:
            self.direction_num = 2
        else:
            self.direction_num = 1
        for layer_i in range(num_layers):
            # 'sfx' means suffix
            sfx = str(layer_i)
            self._add_common_module(sfx, layer_i)
            if self.dropout > 0:
                self.add_module("dropout" + sfx, Dropout(dropout))
            if bidirectional:
                sfx += "_reverse"
                self._add_common_module(sfx, layer_i)

                # For bidirectional LSTMs, forward and backward are directions
                # 0 and 1 respectively, the backward here represents feed the
                # reverse input sequence to lstm, which is different from the
                # general backward used for derivation.
                # Mark the forward direction as f and backward direction as b,
                # so fb means forward and backward direction
                self.add_module(
                    "cat_fb" + str(layer_i),
                    FloatFunctional(qconfig=self.qconfig),
                )
                self.add_module(
                    "cat_fbhs" + sfx, FloatFunctional(qconfig=self.qconfig)
                )
                self.add_module(
                    "cat_fbcs" + sfx, FloatFunctional(qconfig=self.qconfig)
                )
        self.cat_seq_hs = FloatFunctional(qconfig=self.qconfig)
        self.cat_seq_cs = FloatFunctional(qconfig=self.qconfig)

    def _add_common_module(self, sfx, layer_i):
        from changan_plugin_pytorch.nn.quantized import FloatFunctional

        if self.proj_size > 0:
            hlinear_in_features = self.proj_size
        else:
            hlinear_in_features = self.hidden_size
        if layer_i > 0:
            if self.proj_size > 0:
                ilinear_in_features = self.direction_num * self.proj_size
            else:
                ilinear_in_features = self.direction_num * self.hidden_size
        else:
            ilinear_in_features = self.input_size
        self.add_module(
            ("ilinear" + sfx),
            Linear(
                in_features=ilinear_in_features,
                out_features=4 * self.hidden_size,
                bias=self.bias,
                qconfig=self.qconfig,
            ),
        )
        self.add_module(
            "hlinear" + sfx,
            Linear(
                in_features=hlinear_in_features,
                out_features=4 * self.hidden_size,
                bias=self.bias,
                qconfig=self.qconfig,
            ),
        )
        self.add_module(
            "i2h_h2h_add" + sfx, FloatFunctional(qconfig=self.qconfig)
        )
        self.add_module(
            "fg_cell_state_mul" + sfx, FloatFunctional(qconfig=self.qconfig)
        )
        self.add_module(
            "ig_it_mul" + sfx, FloatFunctional(qconfig=self.qconfig)
        )
        self.add_module(
            "next_state_add" + sfx, FloatFunctional(qconfig=self.qconfig)
        )
        self.add_module(
            "next_hid_state_mul" + sfx, FloatFunctional(qconfig=self.qconfig)
        )
        self.add_module(
            "in_gate_sigmoid" + sfx,
            SegmentLUT(torch.sigmoid, False, None, qconfig=self.qconfig),
        )
        self.add_module(
            "f_gate_sigmoid" + sfx,
            SegmentLUT(torch.sigmoid, False, None, qconfig=self.qconfig),
        )
        self.add_module(
            "in_transform_tanh" + sfx,
            SegmentLUT(
                torch.tanh,
                False,
                None,
                qconfig=self.qconfig,
            ),
        )
        self.add_module(
            "out_gate_sigmoid" + sfx,
            SegmentLUT(
                torch.sigmoid,
                False,
                None,
                qconfig=self.qconfig,
            ),
        )
        self.add_module(
            "next_state_tanh" + sfx,
            SegmentLUT(
                torch.tanh,
                False,
                None,
                qconfig=self.qconfig,
            ),
        )
        self.add_module("cap" + sfx, FloatFunctional(qconfig=self.qconfig))
        if self.proj_size > 0:
            self.add_module(
                "proj_linear" + sfx,
                Linear(
                    in_features=self.hidden_size,
                    out_features=self.proj_size,
                    bias=False,
                    qconfig=self.qconfig,
                ),
            )

    def _set_params_impl(self, layer_i, float_lstm, sfx=""):
        sfx = str(layer_i) + sfx
        # 'l' means layer in weight_xx_l or bias_xx_l
        getattr(self, "ilinear" + sfx).weight = float_lstm.get_parameter(
            "weight_ih_l" + sfx
        )
        getattr(self, "hlinear" + sfx).weight = float_lstm.get_parameter(
            "weight_hh_l" + sfx
        )
        if float_lstm.bias:
            getattr(self, "ilinear" + sfx).bias = float_lstm.get_parameter(
                "bias_ih_l" + sfx
            )
            getattr(self, "hlinear" + sfx).bias = float_lstm.get_parameter(
                "bias_hh_l" + sfx
            )
        if self.proj_size > 0:
            getattr(
                self, "proj_linear" + sfx
            ).weight = float_lstm.get_parameter("weight_hr_l" + sfx)

    def _set_params(self, float_lstm):
        for layer_i in range(self.num_layers):
            self._set_params_impl(layer_i, float_lstm)
            if self.bidirectional:
                self._set_params_impl(layer_i, float_lstm, "_reverse")

    def _lstm_cell(self, i2h, hid_state, cell_state, sfx):
        h2h = getattr(self, "hlinear" + sfx)(hid_state)
        gate = getattr(self, "i2h_h2h_add" + sfx).add(i2h, h2h)
        split_gates = torch.split(
            gate, split_size_or_sections=self.hidden_size, dim=1
        )
        in_gate = getattr(self, "in_gate_sigmoid" + sfx)(split_gates[0])
        f_gate = getattr(self, "f_gate_sigmoid" + sfx)(split_gates[1])
        in_transform = getattr(self, "in_transform_tanh" + sfx)(split_gates[2])
        out_gate = getattr(self, "out_gate_sigmoid" + sfx)(split_gates[3])
        next_cell_state = getattr(self, "next_state_add" + sfx).add(
            getattr(self, "fg_cell_state_mul" + sfx).mul(f_gate, cell_state),
            getattr(self, "ig_it_mul" + sfx).mul(in_gate, in_transform),
        )
        next_hid_state = getattr(self, "next_hid_state_mul" + sfx).mul(
            out_gate, getattr(self, "next_state_tanh" + sfx)(next_cell_state)
        )
        if self.proj_size > 0:
            next_hid_state = getattr(self, "proj_linear" + sfx)(next_hid_state)
        return next_hid_state, next_cell_state

    def _lstm(
        self,
        data,
        hid_state,
        cell_state,
        sfx,
        output,
        sequence_length,
        backward=False,
    ):
        # Use the whole sequence input to make a linear, and then split the
        # result along the 0 dimension to replace the linear for each small
        # input in the sequence, so as to improve the efficiency
        total_i2h = getattr(self, "ilinear" + sfx)(data)
        i2hs = torch.split(total_i2h, split_size_or_sections=1, dim=0)
        for seq_i in range(sequence_length):
            # use [0] to change i2h dimention from 3 to 2
            if backward:
                i2h = i2hs[sequence_length - seq_i - 1][0]
            else:
                i2h = i2hs[seq_i][0]
            hid_state, cell_state = self._lstm_cell(
                i2h, hid_state, cell_state, sfx
            )
            if backward:
                output.insert(0, hid_state)
            else:
                output.append(hid_state)
        return hid_state, cell_state

    def forward(self, data, states=None):
        assert_qtensor_input(data)

        # TODO: torch1.11 support unbatched input
        if data.ndim != 3:
            raise RuntimeError(
                "input must have 3 dimensions, got {}".format(data.ndim)
            )
        if self.batch_first:
            data = torch.transpose(data, 1, 0)
        sequence_length = data.shape[0]
        if states is None:
            device = data.as_subclass(torch.Tensor).device
            if isinstance(self.ilinear0, Linear):
                if not hasattr(self, "hid_state"):
                    st_shape = [
                        self.direction_num * self.num_layers,
                        data.shape[1],
                        self.hidden_size,
                    ]
                    hid_shape = st_shape
                    if self.proj_size > 0:
                        hid_shape = [
                            self.direction_num * self.num_layers,
                            data.shape[1],
                            self.proj_size,
                        ]
                    self.hid_state = torch.zeros(hid_shape, device=device)
                    self.cell_state = torch.zeros(st_shape, device=device)
                hid_state = QTensor(
                    self.hid_state, data.q_scale(), self.out_type
                )
                cell_state = QTensor(
                    self.cell_state, data.q_scale(), self.out_type
                )
            else:
                hid_state = QTensor(
                    self.hid_state.to(_storage_type_map[self.out_type]),
                    data.q_scale(),
                    self.out_type,
                )
                cell_state = QTensor(
                    self.cell_state.to(_storage_type_map[self.out_type]),
                    data.q_scale(),
                    self.out_type,
                )
        else:
            hid_state, cell_state = states
        out_hstates = []
        out_cstates = []
        if self.bidirectional:
            for layer_i in range(self.num_layers):
                f_output = []
                b_output = []
                f_hid_state = hid_state[2 * layer_i]
                f_cell_state = cell_state[2 * layer_i]
                b_hid_state = hid_state[2 * layer_i + 1]
                b_cell_state = cell_state[2 * layer_i + 1]
                f_hid_state, f_cell_state = self._lstm(
                    data,
                    f_hid_state,
                    f_cell_state,
                    str(layer_i),
                    f_output,
                    sequence_length,
                )
                b_hid_state, b_cell_state = self._lstm(
                    data,
                    b_hid_state,
                    b_cell_state,
                    str(layer_i) + "_reverse",
                    b_output,
                    sequence_length,
                    backward=True,
                )
                out_hstates.append(
                    getattr(self, "cat_fbhs" + str(layer_i) + "_reverse")
                    .cap([f_hid_state, b_output[0]], dim=0)
                    .reshape(2, f_hid_state.shape[0], f_hid_state.shape[1])
                )
                out_cstates.append(
                    getattr(self, "cat_fbcs" + str(layer_i) + "_reverse")
                    .cap([f_cell_state, b_cell_state], dim=0)
                    .reshape(2, f_cell_state.shape[0], f_cell_state.shape[1])
                )
                f_res = getattr(self, "cap" + str(layer_i)).cap(
                    f_output, dim=0
                )
                f_res = f_res.reshape(
                    len(f_output), f_hid_state.shape[0], f_hid_state.shape[1]
                )
                b_res = getattr(self, "cap" + str(layer_i) + "_reverse").cap(
                    b_output, dim=0
                )
                b_res = b_res.reshape(
                    len(b_output), b_hid_state.shape[0], b_hid_state.shape[1]
                )
                data = getattr(self, "cat_fb" + str(layer_i)).cap(
                    [f_res, b_res], dim=2
                )
                if self.dropout > 0 and layer_i < self.num_layers - 1:
                    data = getattr(self, "dropout" + str(layer_i))(data)
        else:
            for layer_i in range(self.num_layers):
                hstate = hid_state[layer_i]
                cstate = cell_state[layer_i]
                seq_out = []
                hstate, cstate = self._lstm(
                    data,
                    hstate,
                    cstate,
                    str(layer_i),
                    seq_out,
                    sequence_length,
                )
                out_hstates.append(hstate)
                out_cstates.append(cstate)
                data = getattr(self, "cap" + str(layer_i)).cap(seq_out, dim=0)
                data = data.reshape(
                    sequence_length, hstate.shape[0], hstate.shape[1]
                )
                if self.dropout > 0 and layer_i < self.num_layers - 1:
                    data = getattr(self, "dropout" + str(layer_i))(data)
        out = data
        hid_state = self.cat_seq_hs.cap(out_hstates, dim=0)
        cell_state = self.cat_seq_cs.cap(out_cstates, dim=0)
        if not self.bidirectional:
            hid_state = hid_state.reshape(
                self.num_layers,
                out_hstates[0].shape[0],
                out_hstates[0].shape[1],
            )
            cell_state = cell_state.reshape(
                self.num_layers,
                out_cstates[0].shape[0],
                out_cstates[0].shape[1],
            )
        if self.batch_first:
            out = torch.transpose(out, 1, 0)
        return out, (hid_state, cell_state)

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
        qat_lstm = cls(
            input_size=mod.input_size,
            hidden_size=mod.hidden_size,
            num_layers=mod.num_layers,
            batch_first=mod.batch_first,
            dropout=mod.dropout,
            bidirectional=mod.bidirectional,
            proj_size=mod.proj_size,
            bias=mod.bias,
            qconfig=qconfig,
        )
        qat_lstm._set_params(mod)
        return qat_lstm
