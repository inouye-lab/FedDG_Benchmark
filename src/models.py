import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from transformers import GPT2LMHeadModel, GPT2Model
from transformers import GPT2Tokenizer
from transformers import BertForSequenceClassification, BertModel
import copy
from collections import OrderedDict
from torch.nn import init
from transformers import DistilBertForSequenceClassification, DistilBertModel


def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class CNN(nn.Module):
    def __init__(self, input_shape, probabilistic=False):
        super(CNN,self).__init__()
        self.n_outputs = 2048
        self.probabilistic = probabilistic
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0],out_channels=16,kernel_size=5,padding=2),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),  # kernel_size, stride
            nn.Conv2d(in_channels=16,out_channels=64,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        if self.probabilistic:
            self.fc = nn.Linear(in_features=7*7*64,out_features=self.n_outputs * 2)
        else:
            self.fc = nn.Linear(in_features=7*7*64,out_features=self.n_outputs)
    def forward(self,x):
        feature=self.fc(self.conv(x).view(x.shape[0], -1))
        return feature

class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self, input_shape, feature_dimension=2048, probabilistic=False):
        super(ResNet, self).__init__()
        self.probabilistic = probabilistic
        # self.network = torchvision.models.resnet18(pretrained=True)
        # self.n_outputs = 512
        self.network = torchvision.models.resnet50(pretrained=True)
        self.n_outputs = feature_dimension

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]
        self.dropout = nn.Dropout(0)
        if probabilistic:
            self.network.fc = nn.Linear(self.network.fc.in_features,self.n_outputs*2)
        else:
            self.network.fc = nn.Linear(self.network.fc.in_features,self.n_outputs)

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class DenseNet(torch.nn.Module):
    def __init__(self, input_shape, feature_dimension=2048, probabilistic=False, pretrained=True):
        super(DenseNet, self).__init__()
        self.probabilistic = probabilistic

        self.network = torchvision.models.densenet121(pretrained=pretrained)
        self.n_outputs = feature_dimension

        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        self.dropout = nn.Dropout(0)
        if probabilistic:
            self.network.classifier = nn.Linear(self.network.classifier.in_features,self.n_outputs*2)
        else:
            self.network.classifier = nn.Linear(self.network.classifier.in_features,self.n_outputs)

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    # def train(self, mode=True):
    #     """
    #     Override the default train() to freeze the BN parameters
    #     """
    #     super().train(mode)
    #     self.freeze_bn()

    # def freeze_bn(self):
    #     for m in self.network.modules():
    #         if isinstance(m, nn.BatchNorm2d):
    #             m.eval()

class GPT2LMHeadLogit(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.vocab_size

    def __call__(self, x):
        outputs = super().__call__(x)
        logits = outputs[0] #[batch_size, seqlen, vocab_size]
        return logits


class GPT2Featurizer(GPT2Model):
    def __init__(self, config, probabilistic=False):
        self.probabilistic=probabilistic
        super().__init__(config)

    def init_probablistic(self):
        d = self.embed_dim
        self.lm = nn.Linear(in_features=d, out_features=2*d)
        weight_init = torch.cat((torch.eye(d),torch.eye(d)), dim=0)
        weight_init = nn.parameter.Parameter(weight_init, requires_grad=True)
        self.lm.weight=weight_init            

    @property
    def n_outputs(self):
        return self.embed_dim

    def __call__(self, x):
        outputs = super().__call__(x)
        hidden_states = outputs[0] #[batch_size, seqlen, n_embd]
        if self.probabilistic:
            hidden_states = self.lm(hidden_states)
        return hidden_states


class GPT2FeaturizerLMHeadLogit(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.vocab_size
        self.transformer = GPT2Featurizer(config)

    def __call__(self, x):
        hidden_states = self.transformer(x) #[batch_size, seqlen, n_embd]
        logits = self.lm_head(hidden_states) #[batch_size, seqlen, vocab_size]
        return logits


class GeneDistrNet(nn.Module):
    def __init__(self, num_labels, input_size, hidden_size=4096):
        super(GeneDistrNet,self).__init__()
        self.num_labels = num_labels
        self.input_size = input_size
        self.latent_size = 4096
        self.genedistri = nn.Sequential(
            nn.Linear(input_size + self.num_labels, self.latent_size),
            nn.LeakyReLU(),
            nn.Linear(self.latent_size, hidden_size),
            nn.ReLU(),
        )
        self.initial_params()

    def initial_params(self):
        for layer in self.modules():
            if isinstance(layer,torch.nn.Linear):
                init.xavier_uniform_(layer.weight, 0.5)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.genedistri(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, hidden_size, num_labels, rp_size=1024):
        super(Discriminator,self).__init__()
        self.features_pro = nn.Sequential(
            nn.Linear(rp_size, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
        self.optimizer = None
        self.projection = nn.Linear(hidden_size+num_labels, rp_size, bias=False)
        with torch.no_grad():
            self.projection.weight.div_(torch.norm(self.projection.weight, keepdim=True))

    def forward(self, y, z):
        feature = z.view(z.size(0), -1)
        feature = torch.cat([feature, y], dim=1)
        feature = self.projection(feature)
        logit = self.features_pro(feature)
        return logit

def code_gpt_py(probabilistic=False):
    name = 'microsoft/CodeGPT-small-py'
    tokenizer = GPT2Tokenizer.from_pretrained(name)
    model = GPT2FeaturizerLMHeadLogit.from_pretrained(name)
    model.resize_token_embeddings(len(tokenizer))
    featurizer = model.transformer
    featurizer.probabilistic = probabilistic
    classifier = model.lm_head
    model = (featurizer, classifier)
    return model


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class BertClassifier(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.num_labels
        
    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        token_type_ids = x[:, :, 2]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0] 
        return outputs

class BertFeaturizer(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        token_type_ids = x[:, :, 2]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[1] # get pooled output
        return outputs


class DistilBertFeaturizer(DistilBertModel):
    def __init__(self, config, probabilistic=False):
        super().__init__(config)
        self.probabilistic = probabilistic
    
    @property
    def d_out(self):
        return 768

    @property
    def n_outputs(self):
        return self.d_out

    def init_probablistic(self):
        d = self.d_out
        self.probabilistic = True
        self.lm = nn.Linear(in_features=d, out_features=2*d)
        weight_init = torch.cat((torch.eye(d),torch.eye(d)), dim=0)
        weight_init = nn.parameter.Parameter(weight_init, requires_grad=True)
        self.lm.weight=weight_init
    
    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        hidden_state = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        pooled_output = hidden_state[:, 0]
        if self.probabilistic:
            pooled_output =self.lm(pooled_output)
        return pooled_output
