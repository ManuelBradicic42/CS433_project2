import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.models as models

import math


class MultiHeadAttention(nn.Module):
    r"""Multi-headed Attention for input Query, Key, Value

    Multi-headed Attention is a module for attention mechanisms which runs through attention in several times in
    parallel, then the multiple outputs are concatenated and linearly transformed

    Args:
        embed_size  (int): Max embedding size
        num_heads   (int): Number of heads in multi-headed attention; Number of splits in the embedding size
        dropout     (float, optional): Percentage of Dropout to be applied in range 0 <= dropout <=1
        batch_dim   (int, optional): The dimension in which batch dimensions is

    """

    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.2, batch_dim: int = 0):
        super(MultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_dim = batch_dim

        self.dropout_layer = nn.Dropout(dropout)

        self.head_size = self.embed_size // self.num_heads

        assert self.head_size * self.num_heads == self.embed_size, "Heads cannot split Embedding size equally"

        self.Q = nn.Linear(self.embed_size, self.embed_size)
        self.K = nn.Linear(self.embed_size, self.embed_size)
        self.V = nn.Linear(self.embed_size, self.embed_size)

        self.linear = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, q, k, v, mask=None):
        if self.batch_dim == 0:
            out = self.batch_0(q, k, v, mask)
        elif self.batch_dim == 1:
            out = self.batch_1(q, k, v, mask)

        return out

    def batch_0(self, q, k, v, mask=None):
        q_batch_size, q_seq_len, q_embed_size = q.size()
        k_batch_size, k_seq_len, k_embed_size = k.size()
        v_batch_size, v_seq_len, v_embed_size = v.size()

        q = self.Q(q).reshape(q_batch_size, q_seq_len, self.num_heads, self.head_size)
        k = self.K(k).reshape(k_batch_size, k_seq_len, self.num_heads, self.head_size)
        v = self.V(v).reshape(v_batch_size, v_seq_len, self.num_heads, self.head_size)

        attention = self.attention(q, k, v, mask=mask)
        concatenated = attention.reshape(v_batch_size, -1, self.embed_size)
        out = self.linear(concatenated)

        return out

    def batch_1(self, q, k, v, mask=None):
        q_seq_len, q_batch_size, q_embed_size = q.size()
        k_seq_len, k_batch_size, k_embed_size = k.size()
        v_seq_len, v_batch_size, v_embed_size = v.size()

        q = self.Q(q).reshape(q_seq_len, q_batch_size, self.num_heads, self.head_size).transpose(0, 1)
        k = self.K(k).reshape(k_seq_len, k_batch_size, self.num_heads, self.head_size).transpose(0, 1)
        v = self.V(v).reshape(v_seq_len, v_batch_size, self.num_heads, self.head_size).transpose(0, 1)

        attention = self.attention(q, k, v, mask=mask)
        concatenated = attention.reshape(-1, v_batch_size, self.embed_size)

        out = self.linear(concatenated)

        return out

    def attention(self, q, k, v, mask=None):
        scores = torch.einsum("bqhe,bkhe->bhqk", [q, k])

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores /= math.sqrt(self.embed_size)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout_layer(scores)
        attention = torch.einsum("bhql,blhd->bqhd", [scores, v])
        return attention


# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    r"""Positional Encoding for Embedded Input

        Positional Encoding with sine and cosine functions of different frequencies

        Args:
            max_len (int, optional): Max length to be encoded
            d_model (int, optional): Embedding size of input
            dropout (float, optional): A probability from 0 to 1 which determines the dropout rate

    """

    def __init__(self, max_len: int = 5000, d_model: int = 300, dropout: float = 0.1, device="cpu"):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:x.size(0), :], requires_grad=False)
        return self.dropout(x)


class Transformer_Encoder(nn.Module):
    r"""Transformer Encoder Layer

        Transformer Encoder Layer consisting of multi-headed attention and a feed forward neural network with residual
        connections and Layer Normalization

        Args:
            embed_size      (int): max embedding size
            num_heads       (int): Number of heads in multi-headed attention
            ff_hidden_size  (int): Number of hidden units in feed forward network
            dropout         (float, optional): A probability from 0 to 1 which determines the dropout rate
            device          (str, optional): Determines which device to use for computation, by default cpu

    """

    def __init__(self, embed_size: int, num_heads: int, ff_hidden_size: int, dropout: float = 0.2, batch_dim: int = 1,
                 device: str = "cpu"):
        super(Transformer_Encoder, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.ff_hidden_size = ff_hidden_size
        self.dropout = dropout
        self.batch_dim = batch_dim
        self.device = device

        self.Norm1 = nn.LayerNorm(self.embed_size)
        self.Norm2 = nn.LayerNorm(self.embed_size)

        self.multi_attention = MultiHeadAttention(self.embed_size,
                                                  self.num_heads,
                                                  self.dropout,
                                                  batch_dim=self.batch_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, self.ff_hidden_size),
            nn.ReLU(),
            nn.Linear(self.ff_hidden_size, self.embed_size)
        )

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x, mask=None):
        attention = self.multi_attention(x, x, x, mask)
        x = self.dropout_layer(self.Norm1(x + attention))
        x = self.dropout_layer(self.Norm2(x + self.feed_forward(x)))
        return x


class Transformer_Decoder(nn.Module):
    r"""Transformer Decoder Layer

        Transformer Decoder Layer consisting of multi-headed self-attention, a feed forward neural network with residual
        connections and Layer Normalization, and multi-headed attention over the output of the encoder

        Args:
            embed_size  (int): Max embedding size
            num_heads   (int): Number of heads in multi-headed attention
            num_ff      (int): Number of hidden units in feed forward network
            dropout     (float, optional): A probability from 0 to 1 which determines the dropout rate
            device      (str, optional): Determines which device to use for computation, by default cpu

    """

    def __init__(self, embed_size: int, num_heads: int, num_ff: int, dropout: float = 0.1, batch_dim: int = 1,
                 device: str = "cpu"):
        super(Transformer_Decoder, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_ff = num_ff
        self.dropout = dropout
        self.batch_dim = batch_dim
        self.device = device

        self.masked_multiheadattention = MultiHeadAttention(self.embed_size, self.num_heads, self.dropout,
                                                            batch_dim=self.batch_dim)
        self.multiheadattention = MultiHeadAttention(self.embed_size, self.num_heads, self.dropout,
                                                     batch_dim=self.batch_dim)

        self.Norm1 = nn.LayerNorm(self.embed_size)
        self.Norm2 = nn.LayerNorm(self.embed_size)
        self.Norm3 = nn.LayerNorm(self.embed_size)

        self.dropout_layer = nn.Dropout(self.dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, self.num_ff),
            nn.ReLU(),
            nn.Linear(self.num_ff, self.embed_size)
        )

    def forward(self, x, y, y_mask=None, x_mask=None):
        attention1 = self.masked_multiheadattention(y, y, y, y_mask)

        y = self.dropout_layer(self.Norm1(y + attention1))

        attention2 = self.multiheadattention(y, x, x)

        x = self.dropout_layer(self.Norm2(y + attention2))

        x = self.dropout_layer(self.Norm3(x + self.feed_forward(x)))

        return x


class Transformer(nn.Module):
    r"""Transformer Model

        Transformer Model that consists of embedding, encoders, decoders,and feed forward networks.
        It is designed to handle sequential data.

        Args:
            s_vocab_size    (int): Sequence vocabulary size
            t_vocab_size    (int): Transformer vocabulary size
            embed_size      (int): Max embedding size
            num_heads       (int): Number of heads in multi-headed attention
            num_ff          (int): Number of feed forward networks
            encode_layers   (int): Number of encoders in Transformer
            decode_layers   (int): Number of decoders in Transformer
            hidden_size     (int): Number of hidden layers
            dropout         (float, optional): A probability from 0 to 1 which determines the dropout rate
            device          (str, optional): Determines which device to use for computation, by default cpu

    """

    def __init__(self, s_vocab_size: int, t_vocab_size: int, embed_size: int, num_heads: int, num_ff: int,
                 encode_layers: int, decode_layers: int, hidden_size: int, dropout: float = 0.2, device: str = "cpu"):
        super(Transformer, self).__init__()

        self.s_vocab_size = s_vocab_size
        self.t_vocab_size = t_vocab_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_ff = num_ff
        self.encoder_num_layers = encode_layers
        self.decoder_num_layers = decode_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = device

        self.dropout_layer = nn.Dropout(self.dropout)

        self.encoder_embed = nn.Embedding(self.s_vocab_size, embed_size)
        self.decoder_embed = nn.Embedding(self.t_vocab_size, embed_size)
        self.encoder_positional_encoding = PositionalEncoding(self.s_vocab_size, self.embed_size, device=device)
        self.decoder_positional_encoding = PositionalEncoding(self.t_vocab_size, self.embed_size, device=device)

        self.encoders = nn.ModuleList([])
        for layer in range(self.encoder_num_layers):
            self.encoders.append(Transformer_Encoder(self.embed_size, self.num_heads, self.hidden_size, dropout))

        self.decoders = nn.ModuleList([])
        for layer in range(self.decoder_num_layers):
            self.decoders.append(Transformer_Decoder(self.embed_size, self.num_heads, self.hidden_size, dropout,
                                                     self.device))

        self.final = nn.Linear(self.embed_size, self.t_vocab_size)

    def forward(self, x, y, mask=None):

        y_mask = self.get_y_mask(y)

        x = self.encoder_embed(x) * math.sqrt(self.embed_size)
        y = self.decoder_embed(y) * math.sqrt(self.embed_size)

        x = self.encoder_positional_encoding(x)
        y = self.decoder_positional_encoding(y)

        for encoder in self.encoders:
            x = encoder(x)

        for decoder in self.decoders:
            y = decoder(x, y, y_mask=y_mask)

        y = self.dropout_layer(self.final(y))

        return y

    def get_y_mask(self, x):
        s, b = x.size()
        return torch.tril(torch.ones((s, s)).expand(b, 1, s, s)).to(self.device)


class Transformer_with_nn(nn.Module):
    def __init__(self, s_vocab_size: int, t_vocab_size: int, embed_size: int, num_head: int, num_ff: int,
                 encode_layers: int, decode_layers: int, dropout: float = 0.2, device: str = "cpu"):
        super(Transformer_with_nn, self).__init__()

        self.s_vocab_size = s_vocab_size
        self.t_vocab_size = t_vocab_size
        self.embed_size = embed_size
        self.num_head = num_head
        self.num_ff = num_ff
        self.encoder_num_layers = encode_layers
        self.decoder_num_layers = decode_layers
        self.dropout = dropout
        self.device = device

        self.encoder_embed = nn.Embedding(self.s_vocab_size, embed_size)
        self.decoder_embed = nn.Embedding(self.t_vocab_size, embed_size)
        self.encoder_positional_encoding = PositionalEncoding(self.s_vocab_size, self.embed_size, device=device)
        self.decoder_positional_encoding = PositionalEncoding(self.t_vocab_size, self.embed_size, device=device)

        self.encoder_layer = nn.TransformerEncoderLayer(self.embed_size, self.num_head, self.num_ff,
                                                        dropout=self.dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, self.encoder_num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(self.embed_size, self.num_head, self.num_ff,
                                                        dropout=self.dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, self.decoder_num_layers)

        self.transformer = nn.Transformer(self.embed_size, self.num_head, self.encoder_num_layers,
                                          self.decoder_num_layers, self.num_ff, self.dropout)

        self.final = nn.Linear(self.embed_size, self.t_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = self.encoder_embed(x) / math.sqrt(self.embed_size)
        y = self.decoder_embed(y) / math.sqrt(self.embed_size)

        x = self.encoder_positional_encoding(x)
        y = self.decoder_positional_encoding(y)

        x = self.softmax(self.transformer(x, y))

        # memory = self.encoder(x)

        # out = self.decoder(y, memory)

        # x = self.final(out)
        # x = self.softmax(x)

        return x


class VisionEncoder(nn.Module):
    r"""Vision Encoder Model

        An Encoder Layer with the added functionality to encode important local structures of a tokenized image

        Args:
            embed_size      (int): Embedding Size of Input
            num_heads       (int): Number of heads in multi-headed attention
            hidden_size     (int): Number of hidden layers
            dropout         (float, optional): A probability from 0 to 1 which determines the dropout rate

    """

    def __init__(self, embed_size: int, num_heads: int, hidden_size: int, dropout: float = 0.1):
        super(VisionEncoder, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.attention = MultiHeadAttention(self.embed_size, self.num_heads, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_size, 4 * self.embed_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(4 * self.embed_size, self.embed_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attention(x, x, x)
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    r"""Vision Transformer Model

        A transformer model to solve vision tasks by treating images as sequences of tokens.

        Args:
            image_size      (int): Size of input image
            channel_size    (int): Size of the channel
            patch_size      (int): Max patch size, determines number of split images/patches and token size
            embed_size      (int): Embedding size of input
            num_heads       (int): Number of heads in Multi-Headed Attention
            classes         (int): Number of classes for classification of data
            hidden_size     (int): Number of hidden layers
            dropout         (float, optional): A probability from 0 to 1 which determines the dropout rate

    """

    def __init__(self, image_size: int, channel_size: int, patch_size: int, embed_size: int, num_heads: int,
                 classes: int, num_layers: int, hidden_size: int, dropout: float = 0.1, max_length: int = 7):
        super(ViT, self).__init__()

        self.p = patch_size
        self.image_size = image_size
        self.embed_size = embed_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = channel_size * (patch_size ** 2)
        self.num_heads = num_heads
        self.classes = classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.embeddings = nn.Linear(self.patch_size, self.embed_size)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_size))
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.embed_size))

        self.encoders = nn.ModuleList([])
        for layer in range(self.num_layers):
            self.encoders.append(VisionEncoder(self.embed_size, self.num_heads, self.hidden_size, self.dropout))

        self.norm = nn.LayerNorm(self.embed_size)

        # self.classifier = nn.Sequential(
        #     nn.Linear(self.embed_size, self.classes)
        # )
        # List of classifiers
        self.classifiers = nn.ModuleList([
            nn.Linear(self.embed_size, self.classes) for _ in range(max_length)
        ])


    def forward(self, x, mask=None):
        b, c, h, w = x.size()

        x = x.reshape(b, int((h / self.p) * (w / self.p)), c * self.p * self.p)
        x = self.embeddings(x)

        b, n, e = x.size()

        class_token = self.class_token.expand(b, 1, e)
        x = torch.cat((x, class_token), dim=1)
        x = self.dropout_layer(x + self.positional_encoding)

        for encoder in self.encoders:
            x = encoder(x)

        x = x[:, -1, :]

        logits = torch.stack([classifier(x) for classifier in self.classifiers], dim=1)
        
        return logits


class VGG16_classifier(nn.Module):
    def __init__(self, classes, hidden_size, img_size_preprocess=224, preprocess_flag=False, dropout=0.1):
        super(VGG16_classifier, self).__init__()

        self.classes = classes
        self.hidden_size = hidden_size
        self.img_size_preprocess = img_size_preprocess
        self.preprocess_flag = preprocess_flag
        self.dropout = dropout

        self.vgg16 = models.vgg16(pretrained=True)

        for parameter in self.vgg16.parameters():
            parameter.requires_grad = True

        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(self.img_size_preprocess, self.img_size_preprocess)),
            torchvision.transforms.ToTensor()
        ])

        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, self.hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 4, self.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.classes)
        )

    def forward(self, x):
        if self.preprocess_flag:
            x = self.preprocess(x)
        x = self.vgg16(x)
        return x


class DeiT(nn.Module):
    r"""Data-efficient image Transformer (DeiT) Implementation

        The Data-efficient image Transformer (DeiT) is for multi-class image classification which is trained through
        data distillation

        Args:
            image_size      (int): Input Image height/width size
            channel_size    (int): Number of Channels in Input Image
            patch_size      (int): Size of Each Patch for Input Image
            embed_size      (int): Max embedding size
            num_heads       (int): Number of heads in multi-headed attention
            classes         (int): Number in of distinct classes for classification
            num_layers      (int): Number of encoder blocks in DeiT
            hidden_size     (int): Number of hidden units in feed forward of encoder
            teacher_model   (object): Teacher model for data distillation
            dropout         (float, optional): A probability from 0 to 1 which determines the dropout rate

    """

    def __init__(self, image_size: int, channel_size: int, patch_size: int, embed_size: int, num_heads: int,
                 classes: int, num_layers: int, hidden_size: int, teacher_model, dropout: float = 0.1):
        super(DeiT, self).__init__()

        self.image_size = image_size
        self.channel_size = channel_size
        self.p = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = channel_size * (patch_size ** 2)
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.classes = classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.dropout_layer = nn.Dropout(self.dropout)

        self.norm = nn.LayerNorm(self.embed_size)

        self.embeddings = nn.Linear(self.patch_size, self.embed_size)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_size))
        self.distillation_token = nn.Parameter(torch.randn(1, 1, self.embed_size))
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches + 2, self.embed_size))

        self.teacher_model = teacher_model
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad = False
        self.teacher_model.eval()

        self.encoders = nn.ModuleList([])
        for layer in range(self.num_layers):
            self.encoders.append(VisionEncoder(self.embed_size, self.num_heads, self.hidden_size, self.dropout))

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_size, self.classes)
        )

    def forward(self, x, mask=None):
        b, c, h, w = x.size()

        teacher_logits_vector = self.teacher_model(x)

        x = x.reshape(b, int((h / self.p) * (w / self.p)), c * self.p * self.p)
        x = self.embeddings(x)

        b, n, e = x.size()

        class_token = self.class_token.expand(b, 1, e)
        x = torch.cat((class_token, x), dim=1)

        distillation_token = self.class_token.expand(b, 1, e)
        x = torch.cat((x, distillation_token), dim=1)

        x = self.dropout_layer(x + self.positional_encoding)

        for encoder in self.encoders:
            x = encoder(x)

        x, distillation_token = x[:, 0, :], x[:, -1, :]

        x = self.classifier(self.norm(x))

        return x, teacher_logits_vector


class BERT(nn.Module):
    r"""Bidirectional Encoder Representations from Transformers (BERT) Implementation

            The Bidirectional Encoder Representations from Transformers (BERT) is for binary/multi-class image
            classification once pre-trained

            Args:
                vocab_size      (int): Transformer Vocabulary Size
                classes         (int): Number in of distinct classes for classification
                embed_size      (int): Max embedding size
                num_layers      (int): Number of encoder blocks in BERT
                num_heads       (int): Number of heads in multi-headed attention
                hidden_size     (int): Number of hidden units in feed forward of encoder
                dropout         (float, optional): A probability from 0 to 1 which determines the dropout rate

    """

    def __init__(self, vocab_size: int, classes: int, embed_size: int, num_layers: int, num_heads: int, hidden_size: int,
                 dropout: float = 0.2, device="cpu"):
        super(BERT, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.classes = classes

        self.encoder_embed = nn.Embedding(self.vocab_size, embed_size)
        self.encoder_positional_encoding = PositionalEncoding(self.vocab_size, self.embed_size, device=device)

        self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_size))

        self.encoders = nn.ModuleList([])
        for layer in range(self.num_layers):
            self.encoders.append(Transformer_Encoder(self.embed_size,
                                                     self.num_heads,
                                                     self.hidden_size,
                                                     self.dropout,
                                                     batch_dim=0))

    def forward(self, x):
        x = self.encoder_embed(x) * math.sqrt(self.embed_size)
        x = self.encoder_positional_encoding(x)

        b, n, e = x.size()

        class_token = self.class_token.expand(b, 1, e)
        x = torch.cat((x, class_token), dim=1)

        for encoder in self.encoders:
            x = encoder(x)

        return x


class GPT(nn.Module):
    r"""Generative Pre-trained Transformer (GPT) Implementation

            The Generative Pre-trained Transformer 3 (GPT-3) is a language model transformer that is pretrained then
            applied for a variety of Natural Language Processing tasks

            Args:
                vocab_size      (int): Transformer Vocabulary Size
                embed_size      (int): Max embedding size
                num_layers      (int): Number of encoder blocks in BERT
                num_heads       (int): Number of heads in multi-headed attention
                hidden_size     (int): Number of hidden units in feed forward of encoder
                dropout         (float, optional): A probability from 0 to 1 which determines the dropout rate
                device          (str, optional): device to perform computations, defaults to CPU

    """

    def __init__(self, vocab_size, embed_size: int, num_layers: int, num_heads: int, hidden_size: int,
                 dropout: float = 0.2, device="cpu"):
        super(GPT, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.device = device

        self.decoder_embed = nn.Embedding(self.vocab_size, embed_size)
        self.decoder_positional_encoding = PositionalEncoding(self.vocab_size, self.embed_size, device=device)

        self.class_token = nn.Parameter(torch.randn(1, 1, self.embed_size))

        self.decoders = nn.ModuleList([])
        for layer in range(self.num_layers):
            self.decoders.append(Transformer_Decoder(self.embed_size,
                                                     self.num_heads,
                                                     self.hidden_size,
                                                     self.dropout,
                                                     batch_dim=0))

    def forward(self, x, y):
        y_mask = self.get_y_mask(y)

        x = self.decoder_embed(x) * math.sqrt(self.embed_size)
        y = self.decoder_embed(y) * math.sqrt(self.embed_size)

        x = self.decoder_positional_encoding(x)
        y = self.decoder_positional_encoding(y)

        b, n, e = x.size()

        class_token = self.class_token.expand(b, 1, e)
        x = torch.cat((x, class_token), dim=1)

        for decoder in self.decoders:
            x = decoder(x, y, y_mask=y_mask)

        return x

    def get_y_mask(self, x):
        b, s = x.size()
        return torch.tril(torch.ones((s, s)).expand(b, 1, s, s)).to(self.device)