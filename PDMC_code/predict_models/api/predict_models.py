from abc import ABC, abstractmethod
from typing import List

import torch


class MLModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
                One-dimensional prediction of ml model for an output interval of [0, 1].

                Shape of input dimension has to be always two-dimensional (e.g., (1, m), (n, m))

                Parameters
                ----------
                x : torch.Tensor
                    Tabular data of shape N x M (N number of instances, M number of features)

                Returns
                -------
                iterable object
                    Ml model prediction for interval [0, 1] with shape N x 1, 1-d
                """
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.training

        return self.forward(x).detach()



class LinearModel(MLModel):
    def __init__(self, dim_input):
        """

        Parameters
        ----------
        dim_input: int > 0
            number of neurons for this layer
        """
        super().__init__()

        # number of input neurons
        self.input_neurons = dim_input

        # Layer
        self.output = torch.nn.Linear(dim_input, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        output = self.output(x).reshape(-1)
        output = self.sigmoid(output)

        return output


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            # layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class NonLinearModel(MLModel):
    def __init__(self, dim_input, mlp_dims: List[int], dropout: float):
        """

        Parameters
        ----------
        dim_input: int > 0
            number of neurons for this layer
        """
        super().__init__()

        # number of input neurons
        self.input_neurons = dim_input

        self.mlp_dims: List[int] = mlp_dims

        self.dropout: float = dropout

        self.mlp: MultiLayerPerceptron = MultiLayerPerceptron(dim_input, mlp_dims, dropout)

        # Layer
        self.output = torch.nn.Linear(mlp_dims[-1], 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        representation: torch.Tensor = self.mlp(x)
        output = self.output(representation).reshape(-1)
        output = self.sigmoid(output)

        return output