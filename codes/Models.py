import os
import numpy as np
from time import time
import random

import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F


# TDA IMPORTS
import gudhi as gd
import gudhi.representations
from gtda.homology import FlagserPersistence

from utility.parser import parse_args
args = parse_args()

def set_seed(seed, reproducibility=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda")

def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix
def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm
def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


def compute_persistence(gr):
    # 0. fix the fraoh
    epsilon = 0.000001
    np.fill_diagonal(gr, 1)

    # remove 1 from outside the diagonal (they are the same node)
    out = [j for i in range(gr.shape[0]) for j in range(i + 1, gr.shape[1]) if
           abs(gr[i, j] - 1) < epsilon]

    gr = np.delete(gr, out, axis=0)
    gr = np.delete(gr, out, axis=1)

    # dissimilarity
    gr = 1 - gr
    gr[abs(gr - 1) < epsilon] = np.inf

    gr = np.abs(gr)

    persistence = FlagserPersistence().fit_transform([gr])[0]

    persistence_0 = persistence[persistence[:, -1] == 0][:, :2]
    persistence_1 = persistence[persistence[:, -1] == 1][:, :2]
    persistence_0_no_inf = np.array([bars for bars in persistence_0 if bars[1] != np.inf])
    persistence_1_no_inf = np.array([bars for bars in persistence_1 if bars[1] != np.inf])
    # Total persistence
    tp_0 = np.sum(
        np.fromiter((interval[1] - interval[0] for interval in persistence_0_no_inf), dtype=np.dtype(np.float64)))
    tp_1 = np.sum(
        np.fromiter((interval[1] - interval[0] for interval in persistence_1_no_inf), dtype=np.dtype(np.float64)))

    # Average lifetime
    al_0 = tp_0 / len(persistence_0_no_inf)
    al_1 = tp_1 / len(persistence_1_no_inf)

    # Standard deviation
    sd_0 = np.std([(start + end) / 2 for start, end in persistence_0_no_inf])
    sd_1 = np.std([(start + end) / 2 for start, end in persistence_1_no_inf])

    return {'Total_persistence': [tp_0, tp_1], 'Average_lifetime': [al_0, al_1], 'Standard deviation': [sd_0, sd_1]}

def difference(original_values, new_graph):
    new_values = compute_persistence(new_graph)
    acc = 0
    # we compute the difference as the distance between means; ignoring if the 2 intervals overlap, *1 if only one, *2 if none

    # Dimension 0:
    m_0 = original_values['Average_lifetime'][0]
    m_1 = new_values['Average_lifetime'][0]

    sd_0 = original_values['Standard deviation'][0]
    sd_1 = new_values['Standard deviation'][0]

    if (m_0 > m_1):
        m_0, m_1 = m_1, m_0
        sd_0, sd_1 = sd_1, sd_0

    acc += abs((m_0 - sd_0) - (m_1 - sd_1))
    acc += abs((m_0 + sd_0) - (m_1 + sd_1))

    # Dimension 1:
    m_0 = original_values['Average_lifetime'][1]
    m_1 = new_values['Average_lifetime'][1]

    sd_0 = original_values['Standard deviation'][1]
    sd_1 = new_values['Standard deviation'][1]

    if (m_0 > m_1):
        m_0, m_1 = m_1, m_0
        sd_0, sd_1 = sd_1, sd_0

    acc += abs((m_0 - sd_0) - (m_1 - sd_1))
    acc += abs((m_0 + sd_0) - (m_1 + sd_1))

    return acc


def compute_graph_tda(graph, num_landscapes = 10, points_per_landscape = 100, resolution = 100):
    # 0. fix the graph
    epsilon = 0.000001
    np.fill_diagonal(graph, 1)
    # remove 1 from outside the diagonal
    out = [j for i in range(graph.shape[0]) for j in range(i + 1, graph.shape[1]) if
           abs(graph[i, j] - 1) < epsilon]

    graph = np.delete(graph, out, axis=0)
    graph = np.delete(graph, out, axis=1)

    # dissimilarity
    graph = 1 - graph
    graph[ abs(graph - 1) < epsilon] = np.inf

    graph = np.abs(graph)

    # 1. Persistence
    persistence = FlagserPersistence().fit_transform([graph])
    persistence = persistence[0]

    persistence_0 = persistence[persistence[:, -1] == 0][:, :2]
    persistence_1 = persistence[persistence[:, -1] == 1][:, :2]
    persistence_0_no_inf = np.array([bars for bars in persistence_0 if bars[1] != np.inf])
    persistence_1_no_inf = np.array([bars for bars in persistence_1 if bars[1] != np.inf])

    # Total persistence
    pt_0 = np.sum(
        np.fromiter((interval[1] - interval[0] for interval in persistence_0_no_inf), dtype=np.dtype(np.float64)))
    pt_1 = np.sum(
        np.fromiter((interval[1] - interval[0] for interval in persistence_1_no_inf), dtype=np.dtype(np.float64)))

    # Average lifetime
    al_0 = pt_0 / len(persistence_0_no_inf)
    al_1 = pt_1 / len(persistence_1_no_inf)

    # Standard deviation
    sd_0 = np.std([(start + end) / 2 for start, end in persistence_0_no_inf])
    sd_1 = np.std([(start + end) / 2 for start, end in persistence_1_no_inf])

    # Entropy
    PE = gd.representations.Entropy()
    pe_0 = PE.fit_transform([persistence_0_no_inf])[0][0]
    pe_1 = PE.fit_transform([persistence_1_no_inf])[0][0]

    # Betti curve
    bc = gd.representations.vector_methods.BettiCurve()
    bc_0 = bc(persistence_0_no_inf)
    bc_1 = bc(persistence_1_no_inf)

    # Landscapes
    lc = gd.representations.Landscape(num_landscapes=num_landscapes, resolution=points_per_landscape)

    area_under_lc_0 = np.zeros(num_landscapes)
    area_under_lc_1 = np.zeros(num_landscapes)

    lc_0 = lc(persistence_0_no_inf)
    reshaped_landscapes_0 = lc_0.reshape(num_landscapes, points_per_landscape)
    for i in range(num_landscapes):
        area_under_lc_0[i] = np.trapz(reshaped_landscapes_0[i], dx=1)

    lc_1 = lc(persistence_1_no_inf)
    reshaped_landscapes_1 = lc_1.reshape(num_landscapes, points_per_landscape)
    for i in range(num_landscapes):
        area_under_lc_1[i] = np.trapz(reshaped_landscapes_1[i], dx=1)

    # Shilouettes
    p = 2

    s = gd.representations.Silhouette()
    s2 = gd.representations.Silhouette(weight=lambda x: np.power(x[1] - x[0], p), resolution=resolution)

    s_0 = s(persistence_0_no_inf)
    s2_0 = s2(persistence_0_no_inf)
    area_under_s_0 = np.trapz(s_0, dx=1)
    area_under_s2_0 = np.trapz(s2_0, dx=1)

    s_1 = s(persistence_1_no_inf)
    s2_1 = s2(persistence_1_no_inf)
    area_under_s_1 = np.trapz(s_1, dx=1)
    area_under_s2_1 = np.trapz(s2_1, dx=1)

    tensor = torch.tensor(np.concatenate(
        (np.array([pt_0, pt_1, al_0, al_1, sd_0, sd_1, pe_0, pe_1, area_under_s_0, area_under_s_1, area_under_s2_0,
                   area_under_s2_1]), area_under_lc_0, area_under_lc_1, np.array(bc_0), np.array(bc_1))), requires_grad=False)

    # normalize before returning
    return tensor.div(torch.norm(tensor, p=2, dim=-1, keepdim=True))


class LATTICE(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats):
        super().__init__()
        set_seed(args.seed)
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size

        # A continuació només els guarda amb una mica de gràcia per poder accedir per index de manera eficient
        self.user_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # per defecte es lightgcn
        if args.cf_model == 'ngcf':
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            for i in range(self.n_ui_layers):
                self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.dropout_list.append(nn.Dropout(dropout_list[i]))

        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)

        # Latent structure learning 1. initial k-nn modality-aware graphs
        image_adj = build_sim(self.image_embedding.weight.detach())
        # torch.save(image_adj, '../data/%s/%s-core/image_1.pt' % (args.dataset, args.core))
        image_adj = build_knn_neighbourhood(image_adj, topk=args.topk)
        # torch.save(image_adj, '../data/%s/%s-core/image_2.pt' % (args.dataset, args.core))
        image_adj = compute_normalized_laplacian(image_adj)
        # torch.save(image_adj, '../data/%s/%s-core/image_3.pt' % (args.dataset, args.core))
        # torch.save(image_adj, '../data/%s/%s-core/image_adj_%d.pt' % (args.dataset, args.core, args.topk))

        text_adj = build_sim(self.text_embedding.weight.detach())
        # torch.save(text_adj, '../data/%s/%s-core/text_1.pt' % (args.dataset, args.core))
        text_adj = build_knn_neighbourhood(text_adj, topk=args.topk)
        torch.save(text_adj, '../data/%s/%s-core/text_2.pt' % (args.dataset, args.core))
        text_adj = compute_normalized_laplacian(text_adj)
        # torch.save(text_adj, '../data/%s/%s-core/text_3.pt' % (args.dataset, args.core))

        # torch.save(text_adj, '../data/%s/%s-core/text_adj_%d.pt' % (args.dataset, args.core, args.topk))

        self.text_original_adj = text_adj.to(device)
        self.image_original_adj = image_adj.to(device)

        self.image_trs = nn.Linear(image_feats.shape[1], args.feat_embed_dim)
        self.text_trs = nn.Linear(text_feats.shape[1], args.feat_embed_dim)

        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)


    def forward(self, adj, build_item_graph=False):
        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)

        if build_item_graph:
            weight = self.softmax(self.modal_weight)
            self.image_adj = build_sim(image_feats)
            self.image_adj = build_knn_neighbourhood(self.image_adj, topk=args.topk)

            self.text_adj = build_sim(text_feats)
            self.text_adj = build_knn_neighbourhood(self.text_adj, topk=args.topk)

            learned_adj = weight[0] * self.image_adj + weight[1] * self.text_adj
            learned_adj = compute_normalized_laplacian(learned_adj)
            original_adj = weight[0] * self.image_original_adj + weight[1] * self.text_original_adj
            self.item_adj = (1 - args.lambda_coeff) * learned_adj + args.lambda_coeff * original_adj

        else:
            self.item_adj = self.item_adj.detach()

        h = self.item_id_embedding.weight
        for i in range(args.n_layers):
            # producte de matrius
            h = torch.mm(self.item_adj, h)

        if args.cf_model == 'ngcf':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)

                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings
        elif args.cf_model == 'lightgcn':
            # concadena
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)

            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings
        elif args.cf_model == 'mf':
            return self.user_embedding.weight, self.item_id_embedding.weight + F.normalize(h, p=2, dim=1)

class LATTICE_TDA_first_graph(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats):
        super().__init__()
        set_seed(args.seed)
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size

        # A continuació només els guarda amb una mica de gràcia per poder accedir per index de manera eficient
        self.user_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)


        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # per defecte es lightgcn
        if args.cf_model == 'ngcf':
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            for i in range(self.n_ui_layers):
                self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
                self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
                self.dropout_list.append(nn.Dropout(dropout_list[i]))


        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)
            


        image_adj = build_sim(self.image_embedding.weight.detach())
        # torch.save(image_adj, '../data/%s/%s-core/image_1.pt' % (args.dataset, args.core))
        image_adj = build_knn_neighbourhood(image_adj, topk=args.topk)

        image_2 = image_adj.numpy().copy()
        #torch.save(image_adj, '../data/%s/%s-core/image_2.pt' % (args.dataset, args.core))
        image_adj = compute_normalized_laplacian(image_adj)
        # torch.save(image_adj, '../data/%s/%s-core/image_3.pt' % (args.dataset, args.core))
        # torch.save(image_adj, '../data/%s/%s-core/image_adj_%d.pt'%(args.dataset, args.core, args.topk))

        text_adj = build_sim(self.text_embedding.weight.detach())
        # torch.save(text_adj, '../data/%s/%s-core/text_1.pt' % (args.dataset, args.core))
        text_adj = build_knn_neighbourhood(text_adj, topk=args.topk)
        text_2 = text_adj.numpy().copy()
        # torch.save(text_adj, '../data/%s/%s-core/text_2.pt' % (args.dataset, args.core))
        text_adj = compute_normalized_laplacian(text_adj)
        # torch.save(text_adj, '../data/%s/%s-core/text_3.pt' % (args.dataset, args.core))

        # torch.save(text_adj, '../data/%s/%s-core/text_adj_%d.pt'%(args.dataset, args.core, args.topk))


        # 1. Calcular TDA de self.tda_adj = calcular tda (text_adj)

        tda_image = compute_graph_tda(image_2)
        tda_text = compute_graph_tda(text_2)
        self.tda_separated = torch.cat((tda_image, tda_text))

        self.tda_drop = nn.Dropout(0.1)
        # self.tda_drop = nn.Identity()
        # Capa lineal sobre all_embeddings + descriptores
        self.separated_projection = nn.Linear(args.feat_embed_dim + len(self.tda_separated), args.feat_embed_dim)

        self.text_original_adj = text_adj.to(device)
        self.image_original_adj = image_adj.to(device)

        self.image_trs = nn.Linear(image_feats.shape[1], args.feat_embed_dim)
        self.text_trs = nn.Linear(text_feats.shape[1], args.feat_embed_dim)

        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)


    def forward(self, adj, build_item_graph=False):
        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)


        if build_item_graph:
            weight = self.softmax(self.modal_weight)
            self.image_adj = build_sim(image_feats)
            self.image_adj = build_knn_neighbourhood(self.image_adj, topk=args.topk)

            self.text_adj = build_sim(text_feats)
            self.text_adj = build_knn_neighbourhood(self.text_adj, topk=args.topk)

            learned_adj = weight[0] * self.image_adj + weight[1] * self.text_adj
            learned_adj = compute_normalized_laplacian(learned_adj)
            original_adj = weight[0] * self.image_original_adj + weight[1] * self.text_original_adj
            self.item_adj = (1 - args.lambda_coeff) * learned_adj + args.lambda_coeff * original_adj

        else:
            self.item_adj = self.item_adj.detach()

        h = self.item_id_embedding.weight
        for i in range(args.n_layers):
            #producte de matrius
            h = torch.mm(self.item_adj, h)

        if args.cf_model == 'lightgcn':
            #concadena
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)

            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)


            # concatena y contextualiza

            concat = self.tda_drop(self.tda_separated.repeat(i_g_embeddings.size(0),1))
            together = torch.cat((i_g_embeddings, concat), dim=1)
            together = together.to(torch.float32)

            i_g_embeddings = self.separated_projection(together)

            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings
        else:
            raise Exception("Invalid parameter; only lightgcn is prepared for TDA")

class LATTICE_TDA_each_graph(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats):
        super().__init__()
        set_seed(args.seed)
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size

        # A continuació només els guarda amb una mica de gràcia per poder accedir per index de manera eficient
        self.user_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # per defecte es lightgcn
        if args.cf_model == 'ngcf':
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            for i in range(self.n_ui_layers):
                self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.dropout_list.append(nn.Dropout(dropout_list[i]))

        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)

        # Latent structure learning 1. initial k-nn modality-aware graphs
        image_adj = build_sim(self.image_embedding.weight.detach())
        image_adj = build_knn_neighbourhood(image_adj, topk=args.topk)
        image_adj = compute_normalized_laplacian(image_adj)

        text_adj = build_sim(self.text_embedding.weight.detach())
        text_adj = build_knn_neighbourhood(text_adj, topk=args.topk)
        text_adj = compute_normalized_laplacian(text_adj)

        self.separated_projection = nn.Linear(args.feat_embed_dim + 464, args.feat_embed_dim)

        self.text_original_adj = text_adj.to(device)
        self.image_original_adj = image_adj.to(device)

        self.image_trs = nn.Linear(image_feats.shape[1], args.feat_embed_dim)
        self.text_trs = nn.Linear(text_feats.shape[1], args.feat_embed_dim)

        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)


    def forward(self, adj, build_item_graph=False):
        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)

        if build_item_graph:
            weight = self.softmax(self.modal_weight)
            self.image_adj = build_sim(image_feats)
            self.image_adj = build_knn_neighbourhood(self.image_adj, topk=args.topk)

            self.text_adj = build_sim(text_feats)
            self.text_adj = build_knn_neighbourhood(self.text_adj, topk=args.topk)


            tda_image = compute_graph_tda(self.image_adj.detach().numpy())
            tda_text = compute_graph_tda(self.image_adj.detach().numpy())
            self.tda_separated = torch.cat((tda_image, tda_text))

            learned_adj = weight[0] * self.image_adj + weight[1] * self.text_adj
            learned_adj = compute_normalized_laplacian(learned_adj)
            original_adj = weight[0] * self.image_original_adj + weight[1] * self.text_original_adj
            self.item_adj = (1 - args.lambda_coeff) * learned_adj + args.lambda_coeff * original_adj

        else:
            self.item_adj = self.item_adj.detach()

        h = self.item_id_embedding.weight
        for i in range(args.n_layers):
            # producte de matrius
            h = torch.mm(self.item_adj, h)

        if args.cf_model == 'lightgcn':
            # concadena
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)

            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)


            # concatena y contextualiza

            concat = self.tda_separated.repeat(i_g_embeddings.size(0),1)
            together = torch.cat((i_g_embeddings, concat), dim=1)
            together = together.to(torch.float32)
            i_g_embeddings = self.separated_projection(together)

            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings
        else:
            raise Exception("Invalid parameter; only lightgcn is prepared for TDA")


class LATTICE_TDA_drop_nodes(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats, norm_adj,
                 data_generator, percent_nodes_dropped):
        super().__init__()
        set_seed(args.seed)
        p = image_feats.shape[0] * percent_nodes_dropped // 100

        self.n_users = n_users
        self.n_items = n_items - p
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size

        # A continuació només els guarda amb una mica de gràcia per poder accedir per index de manera eficient
        self.user_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # per defecte es lightgcn
        if args.cf_model == 'ngcf':
            self.GC_Linear_list = nn.ModuleList()
            self.Bi_Linear_list = nn.ModuleList()
            self.dropout_list = nn.ModuleList()
            for i in range(self.n_ui_layers):
                self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i + 1]))
                self.dropout_list.append(nn.Dropout(dropout_list[i]))


        # Latent structure learning 1. initial k-nn modality-aware graphs

        image_adj = build_sim(torch.Tensor(image_feats))
        image_adj = build_knn_neighbourhood(image_adj, topk=args.topk)
        image_2 = image_adj.numpy()

        text_adj = build_sim(torch.Tensor(text_feats))
        text_adj = build_knn_neighbourhood(text_adj, topk=args.topk)
        text_2 = text_adj.numpy()
        ###########

        # 1. Calcular TDA original
        original_values_image = compute_persistence(image_2)
        original_values_text = compute_persistence(text_2)

        # remove 25% of nodes, 10 tries, and keep the best
        smallest_difference = None
        best_out = None
        print("p", p)
        for _ in range(10):

            aux_img = image_2.copy()
            aux_text = image_2.copy()

            empty_users = True
            out = []
            tries = 0
            while(empty_users or len(out) < p):

                aux = random.randint(0, image_2.shape[0]-1)
                while aux in out:
                    aux = random.randint(0, image_2.shape[0]-1)

                empty_users = any(set(lst).issubset(set(out + [aux])) for lst in data_generator.train_items.values())
                tries += 1
                if(not empty_users):
                    out.append(aux)
                if tries == 3*p:
                    raise Exception("Max tries exceeded")

            aux_img = np.delete(aux_img, out, axis=0)
            aux_img = np.delete(aux_img, out, axis=1)
            aux_text = np.delete(aux_text, out, axis=0)
            aux_text = np.delete(aux_text, out, axis=1)

            # cannot happen that train_user [u] == []


            diff = difference(original_values_image, aux_img) + 0.5*difference(original_values_text, aux_text)

            if smallest_difference == None or diff < smallest_difference:
                best_img = aux_img.copy()
                best_text = aux_text.copy()
                smallest_difference = diff
                best_out = out


        # Update everything:
        image_features = np.delete(image_feats, best_out, axis=0)
        text_features = np.delete(text_feats, best_out, axis=0)


        self.norm_adj = norm_adj.to_dense().numpy().copy()
        self.norm_adj = np.delete(self.norm_adj, best_out, axis=0)
        self.norm_adj = np.delete(self.norm_adj, best_out, axis=1)

        data_generator.update(self.n_items, best_out)

        self.norm_adj = torch.from_numpy(self.norm_adj)

        text_adj = torch.from_numpy(best_text)
        image_adj = torch.from_numpy(best_img)
        text_adj = compute_normalized_laplacian(text_adj)

        image_adj = compute_normalized_laplacian(image_adj)

        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_features), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_features), freeze=False)
        ###########

        self.text_original_adj = text_adj.to(device)
        self.image_original_adj = image_adj.to(device)

        self.image_trs = nn.Linear(image_feats.shape[1], args.feat_embed_dim)
        self.text_trs = nn.Linear(text_feats.shape[1], args.feat_embed_dim)

        self.modal_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, adj, build_item_graph=False):

        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)

        if build_item_graph:
            weight = self.softmax(self.modal_weight)
            self.image_adj = build_sim(image_feats)
            self.image_adj = build_knn_neighbourhood(self.image_adj, topk=args.topk)

            self.text_adj = build_sim(text_feats)
            self.text_adj = build_knn_neighbourhood(self.text_adj, topk=args.topk)

            learned_adj = weight[0] * self.image_adj + weight[1] * self.text_adj
            learned_adj = compute_normalized_laplacian(learned_adj)
            original_adj = weight[0] * self.image_original_adj + weight[1] * self.text_original_adj

            self.item_adj = (1 - args.lambda_coeff) * learned_adj + args.lambda_coeff * original_adj
        else:
            self.item_adj = self.item_adj.detach()

        h = self.item_id_embedding.weight
        for i in range(args.n_layers):
            # producte de matrius
            h = torch.mm(self.item_adj, h)
        if args.cf_model == 'ngcf':
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
                bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
                bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
                ego_embeddings = sum_embeddings + bi_embeddings
                ego_embeddings = self.dropout_list[i](ego_embeddings)

                norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
                all_embeddings += [norm_embeddings]

            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings
        elif args.cf_model == 'lightgcn':
            # concadena
            ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)

            all_embeddings = [ego_embeddings]
            for i in range(self.n_ui_layers):
                side_embeddings = torch.sparse.mm(adj, ego_embeddings)
                ego_embeddings = side_embeddings
                all_embeddings += [ego_embeddings]
            all_embeddings = torch.stack(all_embeddings, dim=1)
            all_embeddings = all_embeddings.mean(dim=1, keepdim=False)

            u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
            i_g_embeddings = i_g_embeddings + F.normalize(h, p=2, dim=1)
            return u_g_embeddings, i_g_embeddings
        elif args.cf_model == 'mf':
            return self.user_embedding.weight, self.item_id_embedding.weight + F.normalize(h, p=2, dim=1)


class MF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats=None, text_feats=None):
        super().__init__()
        set_seed(args.seed)
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj, build_item_graph=False):
        return self.user_embedding.weight, self.item_embedding.weight


class NGCF(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats=None, text_feats=None):
        super().__init__()
        set_seed(args.seed)
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.dropout_list = nn.ModuleList()
        self.GC_Linear_list = nn.ModuleList()
        self.Bi_Linear_list = nn.ModuleList()

        self.weight_size = [self.embedding_dim] + self.weight_size
        for i in range(self.n_ui_layers):
            self.GC_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.Bi_Linear_list.append(nn.Linear(self.weight_size[i], self.weight_size[i+1]))
            self.dropout_list.append(nn.Dropout(dropout_list[i]))

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, embedding_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def forward(self, adj, build_item_graph):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            sum_embeddings = F.leaky_relu(self.GC_Linear_list[i](side_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = F.leaky_relu(self.Bi_Linear_list[i](bi_embeddings))
            ego_embeddings = sum_embeddings + bi_embeddings
            ego_embeddings = self.dropout_list[i](ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats=None, text_feats=None):
        super().__init__()
        set_seed(args.seed)
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_ui_layers = len(weight_size)

        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def forward(self, adj, build_item_graph):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        return u_g_embeddings, i_g_embeddings


