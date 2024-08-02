import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool

from tqdm import tqdm


class UniXCoder(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(UniXCoder, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        
    def forward(self, input_ids=None,labels=None):
        
        node_output=self.encoder(input_ids, attention_mask=input_ids.ne(1), output_hidden_states=True, return_dict=True)
        cls = node_output.hidden_states[-1][:,0,:] + node_output.hidden_states[1][:,0,:]
        #print(cls_last.shape)
        
        return cls


class Cross_MultiAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, att_dropout=0.0, aropout=0.0):
        super(Cross_MultiAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads


        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)


    def forward(self, x, comment_tree, pad_mask=None):
        '''
        :param x: [batch_size, c, h, w]
        :param comment_tree: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        b, c, h = x.shape
        batch_size = b

        Q = self.Wq(comment_tree)
        K = self.Wk(x)
        V = self.Wv(x)

        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bnij, bnjd -> bnid', att_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)   # [batch_size, h*w, emb_dim]

        return out, att_weights


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(1 * config.hidden_size, 2 * config.hidden_size)
        self.dense = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, x):
        x = self.linear(x)
        x = self.dropout(x)
        x = self.dense(x)
        # ------------- mlp ----------------------
        x = torch.tanh(x)
        x = self.dropout(x)        
        x = self.out_proj(x)
        
        return x

class GAT(torch.nn.Module): 
    def __init__(self, num_node_features, num_relations, encoder,config,tokenizer,args):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.convsA = torch.nn.ModuleList()
        self.convsB = torch.nn.ModuleList()
        self.convsC = torch.nn.ModuleList()
        self.convsD = torch.nn.ModuleList()

        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
        hid_node_features = 128
        head_num = 2
        for i in range(1):
            if i == 0:
                
                self.convsA.append(GATConv(num_node_features, hid_node_features, heads = head_num, concat = False))
                self.convsB.append(GATConv(num_node_features, hid_node_features, heads = head_num, concat = False))
                self.convsC.append(GATConv(num_node_features, hid_node_features, heads = head_num, concat = False))
                self.convsD.append(GATConv(num_node_features, hid_node_features,  heads = head_num, concat = False))

        self.conv2 = GATConv(hid_node_features * 4, hid_node_features * 2, heads = head_num, concat = False)
        self.conv3 = GATConv(hid_node_features * 2, hid_node_features, heads = head_num, concat = False)
        self.mlp = Sequential(
            Linear(hid_node_features * 2, 8),
            ReLU(),
            Linear(8, 1)
        )

        self.mlp2 = Sequential(
            Linear(768, 8),
            ReLU(),
            Linear(8, 1)
        )

        self.multi_classifier = RobertaClassificationHead(config)
        self.cross_att =Cross_MultiAttention(emb_dim=768, num_heads=4, att_dropout=0.0, aropout=0.0)

    def forward(self, x, edge_index, edge_attr, batch, input_ids=None, cross_input_ids = None, labels=None):
        """
        edgeIndex - [[0, 1, ...], [1, 2, ...]]
        edgeAttr - [[1, 0, 0, 0, 1], ...]
        """
        # 1. Obtain node embeddings.
        # Layer 1
        new_edgeAttr = edge_attr 
        xA = x
        for conv in self.convsA:
            xA = conv(xA, edge_index, new_edgeAttr[:, 0].contiguous().long()).relu()
        xB = x
        for conv in self.convsB:
            xB = conv(xB, edge_index, new_edgeAttr[:, 1].contiguous().long()).relu()
        xC = x
        for conv in self.convsC:
            xC = conv(xC, edge_index, new_edgeAttr[:, 2].contiguous().long()).relu()
        xD = x
        for conv in self.convsD:
            xD = conv(xD, edge_index, new_edgeAttr[:, 3].contiguous().long()).relu()

        x = torch.cat((xA, xB, xC, xD), dim=1)  # concat

        # Layer 2
        x = self.conv2(x, edge_index)
        
        # Layer 3
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        graph_output = x # self.mlp(x)
        x = F.dropout(graph_output, p=0.2, training=self.training)

        graph_logit = self.mlp(x)

        seq_outputs=self.encoder(input_ids, attention_mask=input_ids.ne(1), output_hidden_states=True, return_dict=True)

        cls_last = seq_outputs.hidden_states[-1][:,:,:]

        output = torch.mean(cls_last, dim = 1)

        output = output.squeeze(dim = 1)
        
        logits = self.mlp2(output)

        return torch.sigmoid(logits + graph_logit) , torch.sigmoid(graph_logit), torch.sigmoid(logits)

def PGATTrain(model, trainloader, optimizer_graph, optimizer_seq, scheduler_seq, criterion, cur_epoch, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    lossTrain = 0

    bar = tqdm(trainloader,total=len(trainloader), desc= "Epoch " + str(cur_epoch))

    for _, data in enumerate(bar):    
        optimizer_graph.zero_grad()  # Clear gradients.
        optimizer_seq.zero_grad()
        data.to(device)
        
        out, graph, seq = model.forward(data.x, data.edge_index, data.edge_attr, data.batch, data.seq, data.cross_seq)  # Perform a single forward pass.
        if cur_epoch >= 5:
            loss = torch.log(seq[:,0]+1e-10)*data.y + torch.log((1-seq)[:,0]+1e-10)*(1-data.y)
            loss =- loss.mean()
        else:   
            loss = torch.log(graph[:,0]+1e-10)*data.y + torch.log((1-graph)[:,0]+1e-10)*(1-data.y)
            loss =- loss.mean()

        loss.backward()  # Derive gradients.

        lossTrain += loss.item() * len(data.y)

        if cur_epoch >= 5:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer_seq.step()
            scheduler_seq.step()
        else:  
            optimizer_graph.step()

    lossTrain /= len(trainloader.dataset)

    return model, lossTrain