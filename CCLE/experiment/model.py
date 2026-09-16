import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as fun
from CCLE.experiment.myutils import (
    jaccard_coef,
    torch_corr_x_y, scale_sigmoid_activation_function,
    torch_z_normalized, torch_euclidean_dist,
    normalize_similarity
)


class MultiAttentionNeighbor(nn.Module):
    def __init__(self, feature_dim, attention_dim, device="cpu"):
        super(MultiAttentionNeighbor, self).__init__()
        self.device = device
        self.att_layer = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1),
            nn.Sigmoid()
        )
        self.att_layer.to(device)

    def forward(self, feature_matrix):
        feature_matrix = feature_matrix.to(self.device)
        attention_scores = self.att_layer(feature_matrix).squeeze(-1)

        if feature_matrix.ndim == 1 and attention_scores.ndim == 0:
            attention_scores = attention_scores.unsqueeze(0)
        elif attention_scores.ndim == 0:
            attention_scores = attention_scores.view(1)

        if attention_scores.ndim == 1:
            current_sum_scores = attention_scores.sum()
            attention_scores = attention_scores / (current_sum_scores + 1e-8)
        else:
            current_sum_scores = attention_scores.sum(dim=-1, keepdim=True)
            attention_scores = attention_scores / (current_sum_scores + 1e-8)
        return attention_scores


def reliable_neighbor_filter(similarity_matrix, attention_scores_for_neighbors, topk=10):
    n = similarity_matrix.shape[0]
    filtered_similarity = torch.zeros_like(similarity_matrix)
    for i in range(n):
        k_val = min(topk, n)
        if k_val == 0: continue
        weighted_sim_row = similarity_matrix[i, :] * attention_scores_for_neighbors
        # if n < k_val: k_val = n
        if k_val > 0 and n > 0:
            _, topk_indices = torch.topk(weighted_sim_row, k=k_val)
            filtered_similarity[i, topk_indices] = similarity_matrix[i, topk_indices]
    return filtered_similarity


class ConstructAdjMatrixWithHomogeneous(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    def forward(self, filtered_cell_kernel, filtered_drug_sim, original_cell_drug_adj, enable_homogeneous_graph=True):
        filtered_cell_kernel = filtered_cell_kernel.to(self.device)
        filtered_drug_sim = filtered_drug_sim.to(self.device)
        original_cell_drug_adj = original_cell_drug_adj.to(self.device)

        n_cell = filtered_cell_kernel.shape[0]
        n_drug = filtered_drug_sim.shape[0]

        if original_cell_drug_adj.shape[0] != n_cell or original_cell_drug_adj.shape[1] != n_drug:
            raise ValueError(f"Shape mismatch: original_cell_drug_adj ({original_cell_drug_adj.shape}) "
                             f"vs n_cell ({n_cell}), n_drug ({n_drug})")

        if enable_homogeneous_graph:
            top = torch.cat([filtered_cell_kernel, original_cell_drug_adj], dim=1)
            bottom = torch.cat([original_cell_drug_adj.T, filtered_drug_sim], dim=1)
            adj_matrix = torch.cat([top, bottom], dim=0)
        else:
            adj_matrix = torch.zeros(n_cell + n_drug, n_cell + n_drug, device=self.device)
            adj_matrix[:n_cell, n_cell:] = original_cell_drug_adj
            adj_matrix[n_cell:, :n_cell] = original_cell_drug_adj.T
            # adj_matrix[:n_cell, :n_cell] = torch.eye(n_cell, device=self.device)
            # adj_matrix[n_cell:, n_cell:] = torch.eye(n_drug, device=self.device)


        d_sqrt_inv = torch.pow(torch.sum(adj_matrix, dim=1) + 1e-9, -0.5)
        d_sqrt_inv[torch.isinf(d_sqrt_inv)] = 0.0
        d_matrix = torch.diag(d_sqrt_inv)

        identity = torch.eye(adj_matrix.shape[0], device=self.device)
        if adj_matrix.sum() == 0:
            adj_matrix_hat = identity
        else:
            adj_matrix_hat = identity + d_matrix @ adj_matrix @ d_matrix
        return adj_matrix_hat


class FusionFeature(nn.Module):
    def __init__(self, gene_numpy, cna_numpy, mutation_numpy, sigma,
                 feature_drug_numpy, degree=3, attention_dim=32, device="cpu",
                 enable_static_weights=True, enable_dynamic_weights=True,
                 enable_reliable_neighbor_filter_cell=True, enable_reliable_neighbor_filter_drug=True):
        super().__init__()
        self.device = device
        self.shared_dim = 128
        self.n_cell = gene_numpy.shape[0]
        self.n_drug = feature_drug_numpy.shape[0]

        self.enable_static_weights = enable_static_weights
        self.enable_dynamic_weights = enable_dynamic_weights
        self.enable_reliable_neighbor_filter_cell = enable_reliable_neighbor_filter_cell
        self.enable_reliable_neighbor_filter_drug = enable_reliable_neighbor_filter_drug

        self.gene = torch_z_normalized(torch.from_numpy(gene_numpy).float()).to(device)
        self.cna = torch_z_normalized(torch.from_numpy(cna_numpy).float()).to(device)
        self.mutation = torch.from_numpy(mutation_numpy).float().to(device)
        self.feature_drug = torch.from_numpy(feature_drug_numpy).float().to(device)

        self.gene_mlp = nn.Linear(self.gene.shape[1], self.shared_dim).to(self.device)
        self.cna_mlp = nn.Linear(self.cna.shape[1], self.shared_dim).to(self.device)
        self.mutation_mlp = nn.Linear(self.mutation.shape[1], self.shared_dim).to(self.device)
        self.drug_mlp = nn.Linear(self.feature_drug.shape[1], self.shared_dim).to(self.device)

        self.gene_kernel = self._build_gaussian_kernel(self.gene, sigma)
        self.cna_kernel = self._build_polynomial_kernel(self.cna, degree=degree)
        self.mutation_kernel = jaccard_coef(self.mutation).to(device)
        self.drug_jac_similarity = jaccard_coef(self.feature_drug).to(device)

        if self.enable_static_weights:
            self.weights = nn.Parameter(torch.ones(3, device=device))
        else:
            self.weights = None

        if self.enable_dynamic_weights:
            self.attention_layer = nn.Sequential(
                nn.Linear(self.shared_dim, attention_dim),
                nn.ReLU(),
                nn.Linear(attention_dim, 3),
                nn.Softmax(dim=1)
            ).to(device)
        else:
            self.attention_layer = None

        self.cell_attention = MultiAttentionNeighbor(self.shared_dim, attention_dim, device=device)
        self.drug_attention = MultiAttentionNeighbor(self.shared_dim, attention_dim, device=device)

        self.kernel2feat_cell = nn.Linear(self.n_cell, self.shared_dim).to(device)
        self.kernel2feat_drug = nn.Linear(self.n_drug, self.shared_dim).to(device)

        self.attention_scores_cell_for_GModel = None
        self.filtered_kernel_for_GModel = None
        self.filtered_drug_sim_for_GModel = None

    def _build_gaussian_kernel(self, data, sigma):
        dist = torch_euclidean_dist(data, dim=0)
        return torch.exp(-dist.pow(2) / (2 * sigma ** 2))

    def _build_polynomial_kernel(self, data, degree=2):
        return (data @ data.T + 1) ** degree

    def fusion_cell_feature(self):
        gene_embed = self.gene_mlp(self.gene)
        cna_embed = self.cna_mlp(self.cna)
        mutation_embed = self.mutation_mlp(self.mutation)

        kernels = [self.gene_kernel, self.cna_kernel, self.mutation_kernel]
        kernels_norm = []
        for k_mat in kernels:
            norm_k = torch.norm(k_mat, p='fro')
            kernels_norm.append(k_mat / (norm_k + 1e-8))

        shared_fusion_embed = (gene_embed + cna_embed + mutation_embed) / 3

        if self.enable_static_weights and self.enable_dynamic_weights:
            dynamic_weights = self.attention_layer(shared_fusion_embed)
            static_weights = torch.softmax(self.weights, dim=0)
            fused_kernel = torch.zeros_like(kernels_norm[0])
            for i in range(3):
                mixed_weight_scalar = 0.5 * static_weights[i] + 0.5 * dynamic_weights[:, i].mean()
                fused_kernel += mixed_weight_scalar * kernels_norm[i]
        elif self.enable_static_weights:
            static_weights = torch.softmax(self.weights, dim=0)
            fused_kernel = torch.zeros_like(kernels_norm[0])
            for i in range(3):
                fused_kernel += static_weights[i] * kernels_norm[i]
        elif self.enable_dynamic_weights:
            dynamic_weights = self.attention_layer(shared_fusion_embed)
            fused_kernel = torch.zeros_like(kernels_norm[0])
            for i in range(3):
                fused_kernel += dynamic_weights[:, i].mean() * kernels_norm[i]
        else:
            fused_kernel = (kernels_norm[0] + kernels_norm[1] + kernels_norm[2]) / 3.0


        cell_node_attention_scores = self.cell_attention(shared_fusion_embed)
        if self.enable_reliable_neighbor_filter_cell:
            filtered_kernel = reliable_neighbor_filter(fused_kernel, cell_node_attention_scores, topk=10)
        else:
            filtered_kernel = fused_kernel

        self.attention_scores_cell_for_GModel = cell_node_attention_scores
        self.filtered_kernel_for_GModel = filtered_kernel

        fused_feature = self.kernel2feat_cell(filtered_kernel)
        return fused_feature

    def fusion_drug_feature(self):
        drug_embed = self.drug_mlp(self.feature_drug)
        drug_node_attention_scores = self.drug_attention(drug_embed)
        if self.enable_reliable_neighbor_filter_drug:
            filtered_drug_sim = reliable_neighbor_filter(self.drug_jac_similarity, drug_node_attention_scores, topk=10)
        else:
            filtered_drug_sim = self.drug_jac_similarity
        filtered_drug_sim = normalize_similarity(filtered_drug_sim)

        self.filtered_drug_sim_for_GModel = filtered_drug_sim

        drug_feature = self.kernel2feat_drug(filtered_drug_sim)
        return drug_feature

    def forward(self):
        cell_feature = self.fusion_cell_feature()
        drug_feature = self.fusion_drug_feature()
        fusion_feature = torch.cat([cell_feature, drug_feature], dim=0)
        return fusion_feature


class GEncoder(nn.Module):
    def __init__(self, input_feature_dim, n_hid):
        super().__init__()
        self.lm = nn.Linear(input_feature_dim, n_hid, bias=False)

    def forward(self, feature, adj_mat):
        feature = feature.to(self.lm.weight.device)
        adj_mat = adj_mat.to(self.lm.weight.device)

        input_agg = adj_mat @ feature
        lm_out = fun.relu(self.lm(input_agg))
        return lm_out

    def sim(self, z1, z2):
        z1_norm = fun.normalize(z1, p=2, dim=1)
        z2_norm = fun.normalize(z2, p=2, dim=1)
        return z1_norm @ z2_norm.T


class GDecoder(nn.Module):
    def __init__(self, n_cell, n_drug, n_hid1, n_hid2, alpha):
        super().__init__()
        self.n_cell = n_cell
        self.n_drug = n_drug
        self.alpha = alpha
        self.lm_cell = nn.Linear(n_hid1, n_hid2, bias=False)
        self.lm_drug = nn.Linear(n_hid1, n_hid2, bias=False)

    def forward(self, encode_output):
        encode_output = encode_output.to(self.lm_cell.weight.device)
        z_cell, z_drug = torch.split(encode_output, [self.n_cell, self.n_drug], dim=0)

        cell = fun.normalize(self.lm_cell(z_cell), dim=1)
        drug = fun.normalize(self.lm_drug(z_drug), dim=1)

        output = torch_corr_x_y(cell, drug)
        output = torch.clamp(output, min=-1.0, max=1.0)

        output = scale_sigmoid_activation_function(output, alpha=self.alpha)

        output = torch.clamp(output, min=0.0, max=1.0)

        return output


def calculate_custom_infonce(sim_matrix, pos_mask_float, neg_mask_float, device):
    sim_exp = torch.exp(sim_matrix)
    sim_pos_terms = sim_exp * pos_mask_float
    sim_neg_terms = sim_exp * neg_mask_float

    sum_pos_per_anchor = sim_pos_terms.sum(dim=1)
    sum_neg_per_anchor = sim_neg_terms.sum(dim=1)
    denominator = sum_pos_per_anchor + sum_neg_per_anchor + 1e-9

    valid_anchor_mask = (pos_mask_float.sum(dim=1) > 1e-9) & (denominator > 1e-8)

    if not valid_anchor_mask.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    ratio = sum_pos_per_anchor[valid_anchor_mask] / denominator[valid_anchor_mask]
    log_probs = torch.log(torch.clamp(ratio, min=1e-9))
    loss = -log_probs.mean()
    return loss


class GModel(nn.Module):
    def __init__(self, adj_mat,
                 gene, cna, mutation,
                 sigma,
                 k, iterates,
                 feature_drug,
                 degree,
                 attention_dim,
                 n_hid1,
                 n_hid2,
                 alpha,
                 cl_noise_factor=0.05,
                 cl_edge_drop_prob=0.1,
                 cl_temperature=0.2,
                 device="cpu",
                 enable_static_weights=True,
                 enable_dynamic_weights=True,
                 enable_homogeneous_graph_connections=True,
                 enable_contrastive_learning=True,
                 enable_feature_perturbation=True,
                 enable_structural_perturbation=True,
                 enable_reliable_neighbor_filter_cell=True, # Added to FusionFeature constructor
                 enable_reliable_neighbor_filter_drug=True # Added to FusionFeature constructor
                 ):
        super().__init__()
        self.device = device
        self.n_cell = gene.shape[0]
        self.n_drug = feature_drug.shape[0]

        if isinstance(adj_mat, np.ndarray):
            self.original_cell_drug_adj = torch.from_numpy(adj_mat).float().to(device)
        elif isinstance(adj_mat, torch.Tensor):
            self.original_cell_drug_adj = adj_mat.float().to(device)
        else:
            raise TypeError("adj_mat")

        self.cl_noise_factor = cl_noise_factor
        self.cl_edge_drop_prob = cl_edge_drop_prob
        self.cl_temperature = cl_temperature

        self.enable_homogeneous_graph_connections = enable_homogeneous_graph_connections
        self.enable_contrastive_learning = enable_contrastive_learning
        self.enable_feature_perturbation = enable_feature_perturbation
        self.enable_structural_perturbation = enable_structural_perturbation

        self.fusioner = FusionFeature(
            gene_numpy=gene,
            cna_numpy=cna,
            mutation_numpy=mutation,
            sigma=sigma,
            feature_drug_numpy=feature_drug,
            degree=degree,
            attention_dim=attention_dim,
            device=device,
            enable_static_weights=enable_static_weights,
            enable_dynamic_weights=enable_dynamic_weights,
            enable_reliable_neighbor_filter_cell=enable_reliable_neighbor_filter_cell,
            enable_reliable_neighbor_filter_drug=enable_reliable_neighbor_filter_drug
        )
        self.adj_matrix_constructor = ConstructAdjMatrixWithHomogeneous(device=device)
        self.encoder = GEncoder(input_feature_dim=self.fusioner.shared_dim, n_hid=n_hid1)
        self.encoder.to(device)

        self.decoder = GDecoder(
            self.n_cell, self.n_drug,
            n_hid1=n_hid1, n_hid2=n_hid2, alpha=alpha
        )
        self.decoder.to(device)

    def _create_structurally_perturbed_adj(self, filtered_cell_kernel, filtered_drug_sim):
        adj_cd_perturbed = self.original_cell_drug_adj.clone()
        edge_indices_rows, edge_indices_cols = adj_cd_perturbed.nonzero(as_tuple=True)
        num_edges = len(edge_indices_rows)

        if num_edges > 0:
            num_edges_to_drop = int(num_edges * self.cl_edge_drop_prob)
            if num_edges_to_drop > 0:
                perm = torch.randperm(num_edges, device=self.device)
                drop_indices_in_perm = perm[:num_edges_to_drop]

                selected_row_indices = edge_indices_rows[drop_indices_in_perm]
                selected_col_indices = edge_indices_cols[drop_indices_in_perm]

                adj_cd_perturbed[selected_row_indices, selected_col_indices] = 0

        adj_matrix_hat_struct_aug = self.adj_matrix_constructor(
            filtered_cell_kernel,
            filtered_drug_sim,
            adj_cd_perturbed,
            enable_homogeneous_graph=self.enable_homogeneous_graph_connections
        )
        return adj_matrix_hat_struct_aug.to(self.device)

    def forward(self):

        fused_node_features = self.fusioner().to(self.device)

        current_cell_attention_scores = self.fusioner.attention_scores_cell_for_GModel.to(self.device)
        current_filtered_cell_kernel = self.fusioner.filtered_kernel_for_GModel.to(self.device)
        current_filtered_drug_sim = self.fusioner.filtered_drug_sim_for_GModel.to(self.device)

        adj_matrix_hat_orig = self.adj_matrix_constructor(
            current_filtered_cell_kernel,
            current_filtered_drug_sim,
            self.original_cell_drug_adj,
            enable_homogeneous_graph=self.enable_homogeneous_graph_connections # Control this here
        ).to(self.device)

        encode_output_orig = self.encoder(fused_node_features, adj_matrix_hat_orig)
        output_prediction = self.decoder(encode_output_orig)

        contrastive_loss = torch.tensor(0.0, device=self.device)

        if self.enable_contrastive_learning:
            if self.enable_feature_perturbation:
                encode_output_feat_aug = encode_output_orig + \
                                         self.cl_noise_factor * torch.randn_like(encode_output_orig)
            else:
                encode_output_feat_aug = encode_output_orig

            if self.enable_structural_perturbation:
                adj_matrix_hat_struct_aug = self._create_structurally_perturbed_adj(
                    current_filtered_cell_kernel,
                    current_filtered_drug_sim
                )
                encode_output_struct_aug = self.encoder(fused_node_features, adj_matrix_hat_struct_aug)
            else:
                encode_output_struct_aug = encode_output_orig

            device = encode_output_orig.device
            eye = torch.eye(self.n_cell + self.n_drug, device=device)
            adj_cd_orig_bin = (self.original_cell_drug_adj > 0).float()

            pos_mask_cd_connections = torch.zeros(self.n_cell + self.n_drug, self.n_cell + self.n_drug, device=device)
            pos_mask_cd_connections[:self.n_cell, self.n_cell:] = adj_cd_orig_bin
            pos_mask_cd_connections[self.n_cell:, :self.n_cell] = adj_cd_orig_bin.T

            pos_mask_inter_view_initial = ((eye + pos_mask_cd_connections) > 0).float()
            pos_mask_intra_view_orig_initial = (pos_mask_cd_connections > 0).float()

            attention_scores_cell = current_cell_attention_scores
            reliable_mask = torch.zeros_like(pos_mask_inter_view_initial)
            for i in range(self.n_cell):
                if adj_cd_orig_bin[i, :].sum() == 0: continue
                valid_scores_for_cell_i_drugs = attention_scores_cell[i] * adj_cd_orig_bin[i, :]
                num_actual_drug_connections = int(adj_cd_orig_bin[i, :].sum().item())
                if num_actual_drug_connections == 0: continue
                topk_val = min(10, num_actual_drug_connections)

                if topk_val > 0:
                    _, topk_drug_indices_for_cell_i = torch.topk(valid_scores_for_cell_i_drugs, k=topk_val)
                    for drug_idx_in_block in topk_drug_indices_for_cell_i:
                        actual_drug_node_idx = self.n_cell + drug_idx_in_block
                        reliable_mask[i, actual_drug_node_idx] = 1
                        reliable_mask[actual_drug_node_idx, i] = 1

            reliable_mask_bool = reliable_mask.bool()
            pos_mask_inter_initial_bool = pos_mask_inter_view_initial.bool()
            pos_mask_final_inter_bool = pos_mask_inter_initial_bool & ~reliable_mask_bool
            neg_mask_final_inter_bool = ~(pos_mask_inter_initial_bool | reliable_mask_bool)
            pos_mask_final_inter_float = pos_mask_final_inter_bool.float()
            neg_mask_final_inter_float = neg_mask_final_inter_bool.float()

            pos_mask_intra_orig_initial_bool = pos_mask_intra_view_orig_initial.bool()
            pos_mask_final_intra_orig_bool = pos_mask_intra_orig_initial_bool & ~reliable_mask_bool
            neg_mask_final_intra_orig_bool = ~(pos_mask_intra_orig_initial_bool | reliable_mask_bool | eye.bool())
            pos_mask_final_intra_orig_float = pos_mask_final_intra_orig_bool.float()
            neg_mask_final_intra_orig_float = neg_mask_final_intra_orig_bool.float()

            sim_orig_vs_feat_aug = self.encoder.sim(encode_output_orig, encode_output_feat_aug) / self.cl_temperature
            sim_orig_vs_struct_aug = self.encoder.sim(encode_output_orig, encode_output_struct_aug) / self.cl_temperature
            sim_orig_vs_orig = self.encoder.sim(encode_output_orig, encode_output_orig) / self.cl_temperature

            loss_feat = calculate_custom_infonce(sim_orig_vs_feat_aug, pos_mask_final_inter_float,
                                                 neg_mask_final_inter_float, device)
            loss_struct = calculate_custom_infonce(sim_orig_vs_struct_aug, pos_mask_final_inter_float,
                                                   neg_mask_final_inter_float, device)
            loss_orig_intra = calculate_custom_infonce(sim_orig_vs_orig, pos_mask_final_intra_orig_float,
                                                       neg_mask_final_intra_orig_float, device)

            contrastive_loss = (loss_feat + loss_struct + loss_orig_intra) / 3.0

        return output_prediction, contrastive_loss