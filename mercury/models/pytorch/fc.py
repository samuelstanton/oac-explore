import torch
from torch import Size
from torch.nn import Linear, BatchNorm1d, ReLU
from torch.utils.data import DataLoader
from collections import OrderedDict


class PytorchFC(torch.nn.Sequential):
    """
    Multi-Layer Perceptron
    """

    def __init__(
        self,
        input_shape: Size,
        output_shape: Size,
        hidden_width: int or list,
        depth: int = 4,
        activation: str = 'relu',
        batch_norm: bool = True,
        holdout_ratio: float = 0.2,
        max_epochs_since_update: int = 5,
        minibatch_size: int = 256,
    ) -> None:
        """
        :param input_shape:
        :param output_shape:
        :param hidden_width:
        """
        params = locals()
        del params['self']
        self.__dict__ = params
        if isinstance(hidden_width, list) and len(hidden_width) != depth:
            raise ValueError("hidden width must be an int or a list with len(depth)")
        elif isinstance(hidden_width, int) and depth > 0:
            hidden_width = [hidden_width] * depth
        modules = []
        input_width, output_width = *input_shape, *output_shape
        if depth == 0:
            modules.append(("linear1", Linear(input_width, output_width)))
        else:
            modules.append(("linear1", Linear(input_width, hidden_width[0])))
            for i in range(1, depth + 1):
                if batch_norm:
                    modules.append((f"bn{i}", BatchNorm1d(hidden_width[i-1])))
                if activation == 'relu':
                    modules.append((f"relu{i}", ReLU()))
                elif activation == 'swish':
                    modules.append((f"swish{i}", Swish()))
                else:
                    raise ValueError("Unrecognized activation")
                modules.append(
                    (
                        f"linear{i + 1}",
                        Linear(
                            hidden_width[i-1], hidden_width[i] if i != depth else output_width
                        ),
                    )
                )
        modules = OrderedDict(modules)
        super().__init__(modules)

    def fit(self, dataset, max_epochs=None):
        metrics = {
            'holdout_mse': [],
        }
        n_val = min(int(5e3), int(self.holdout_ratio * len(dataset)))
        assert n_val > 0
        n_train = len(dataset) - n_val
        train_data, val_data = torch.utils.data.random_split(dataset, [n_train, n_val])
        val_x, val_y = val_data[:]

        mse_loss = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.optim_param_groups)
        dataloader = DataLoader(train_data, batch_size=self.minibatch_size, shuffle=True)
        exit_training = False
        epoch = 1
        snapshot = (1, 1e6, self.state_dict())
        while not exit_training:
            self.train()
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                pred = self(inputs)
                loss = mse_loss(pred, labels)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.eval()
                pred = self(val_x)
                holdout_loss = mse_loss(pred, val_y)
                metrics['holdout_mse'].append(holdout_loss)

            snapshot, exit_training = self._save_best(epoch, holdout_loss, snapshot)
            if exit_training or (max_epochs and epoch == max_epochs):
                break
            epoch += 1
        self.load_state_dict(snapshot[2])
        return metrics

    def _save_best(self, epoch, holdout_loss, snapshot):
        exit_training = False
        last_update, best_loss, _ = snapshot
        improvement = (best_loss - holdout_loss) / max(abs(best_loss), 1.)
        if improvement > 0.01:
            snapshot = (epoch, holdout_loss.item(), self.state_dict())
        if epoch == snapshot[0] + self.max_epochs_since_update:
            exit_training = True
        return snapshot, exit_training

    @property
    def optim_param_groups(self):
        weight_decay = [0.000025, 0.00005, 0.000075, 0.000075, 0.0001]
        groups = []
        for m in self.modules():
            if isinstance(m, Linear):
                groups.append(
                    {
                        'params': [m.weight, m.bias],
                        'weight_decay': weight_decay.pop(0)
                    }
                )

        other_params = []
        for name, param in self.named_parameters():
            if 'linear' not in name:
                other_params.append(param)
        groups.append({'params': other_params})

        return groups


class Swish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs * torch.sigmoid(inputs)
