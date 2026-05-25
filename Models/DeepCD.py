import os
import torch
import torch.nn as nn
from Models.Encoders import AttentionEncoder_tree
from Models.TreeDecoder import GCNDecoder
import json

class HiLLM(nn.Module):
    def __init__(self, constants, q_matrix, dataset_name, decoder_config=None):
        super().__init__()


        n_exercise, n_concept = constants[1], constants[2]

        self.concept_tree, self.n_leaf = build_tree_with_id(dataset_name, filed_name=dataset_name)

        self.assess_net = AttentionEncoder_tree(
            n_exercise, n_concept,
            n_concept=n_concept,
            out_sigmoid=False,
            pooling="mean",
            block_num=1,
            q_matrix=q_matrix,
            concept_tree=self.concept_tree
        )

        if decoder_config is None:
            decoder_config = {}

        self.CD_model = GCNDecoder(
            n_exercise=n_exercise,
            n_nodes=self.assess_net.n_node,
            final_tree=self.concept_tree,
            Q=q_matrix,
            **decoder_config
        )

        self.Q_matrix = q_matrix
        self.expanded_Q_matrix = ((q_matrix @ self.assess_net.node_mask.to(torch.float32).T) > 0).to(torch.float32)

    def forward(self, p_matrix, stu_id, **kwargs):
        theta = self.assess_net(p_matrix)
        predict = self.CD_model(theta)
        return predict, theta


def build_tree_with_id(dataset_name, filed_name):
    tree_json_path = os.path.join("../Datasets", dataset_name, "hierarchical_kp_tree.json")
    concept_id_map_path = os.path.join("../Datasets", dataset_name, "concept_id_map.json")
    with open(tree_json_path, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    with open(concept_id_map_path, "r", encoding="utf-8") as f:
        concept_id_map = json.load(f)

    max_leaf_id = max(concept_id_map.values()) if len(concept_id_map) > 0 else -1
    next_internal_id = max_leaf_id + 1

    def process_node(node):
        nonlocal next_internal_id

        label = node.get("label", "")
        children = node.get("children", [])
        items = node.get("items", [])

        if len(children) == 0 and len(items) > 0:
            this_id = next_internal_id
            next_internal_id += 1

            new_children = []
            for item in items:
                if item not in concept_id_map:
                    continue
                leaf_id = concept_id_map[item]
                new_children.append({
                    "id": leaf_id,
                    "name": item,
                    "children": []
                })

            return {
                "id": this_id,
                "name": label,
                "children": new_children
            }

        this_id = next_internal_id
        next_internal_id += 1

        new_children = [process_node(c) for c in children]

        return {
            "id": this_id,
            "name": label,
            "children": new_children
        }

    processed_children = [process_node(root) for root in raw_list]

    top_id = next_internal_id
    next_internal_id += 1

    final_tree = {
        "id": top_id,
        "name": filed_name,
        "children": processed_children
    }

    return final_tree, next_internal_id
