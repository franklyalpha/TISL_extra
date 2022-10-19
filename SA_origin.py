# all implementations should be from scratch, and must solely follow from the paper before finally check
# with online implementations
import torch.nn as nn
import torch


class slot_attention(nn.Module):
    """
    assumptions for using this module:
    the input_shape is assumed to be: (B, D_i) where B is batch, D_i is the input dim;
    slot shape is assumed to be: (K, D_s) where K is number of slots, and D_s is the slot feature dim

    """
    def __init__(self, input_shape, slot_shape, attention_dim,
                 num_slots, slot_mu, slot_sigma):

        super(slot_attention, self).__init__()
        self.input_shape = input_shape
        self.slot_shape = slot_shape
        self.attention_dim = attention_dim
        self.q = nn.Linear(slot_shape, attention_dim)
        self.k = nn.Linear(input_shape, attention_dim)
        self.v = nn.Linear(input_shape, input_shape)

        self.gru = nn.GRUCell(slot_shape, slot_shape)

        self.num_slots = num_slots
        self.slot_mu = slot_mu
        self.slot_sigma = slot_sigma

        self.input_ln = nn.LayerNorm(input_shape)
        self.slots_ln = nn.LayerNorm(slot_shape)

    def forward(self, input):
        """

        :param input: size: [batch, num_input, D], where D is feature dimension
        :return:
        """
        # first generate slots
        slots = self.create_slots(self.slot_mu, self.slot_sigma,
                                  self.num_slots, self.slot_shape, input.shape[0])
        # normalize input using layer norm
        norm_input = self.input_ln(input)
        # perform attention
        query = self.q(slots) # shape [..., K, att]
        key = self.k(input) # shape [..., F, att]
        value = self.v(input) # shape [..., F, input_dim]
        dot_product = query.matmul(key.swapaxes(-1, -2)) * (self.attention_dim ** (-0.5)) # realizing the result is the
                                                                        # same as (key.matmul(query.swapaxes(-1, -2))).swapaxes(-1, -2)
        # alternative implementation:
        # dot_product = torch.einsum("bqa, bka -> bqk", query, key)

        weight = nn.Softmax(dim=-1)(dot_product) # realizing "dot_product" has shape [..., K, F],
                                                # dim=-1 normalizes "inputs" so for each slot the sum is 1
        slot_prediction = weight.matmul(value) # [..., F, input_dim]



    def create_slots(self, mu, sigma, k, d_slots, batch_size):
        """
        :param mu: Gaussian sampling mean
        :param sigma: Gaussian sampling sigma
        :param k: number of slots
        :param d_slots: slot dimension
        :return:
        """
        # will sample slots according to gaussian.
        slots = torch.fill(torch.zeros([batch_size, k, d_slots]), mu)
        return slots + sigma * torch.randn(slots.shape) # this will return sampled slots, each one is different

input = torch.randn([2, 50, 75])
