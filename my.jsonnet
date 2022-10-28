// Configuration for a coreference resolution model based on:
//   Lee, Kenton et al. “Higher-order Coreference Resolution with Coarse-to-fine Inference.” NAACL (2018).
//   + SpanBERT-large

local transformer_model = 'SpanBERT/spanbert-large-cased';
local max_length = 512;
local feature_size = 20;
local max_span_width = 3;

local transformer_dim = 1024;  # uniquely determined by transformer_model
local span_embedding_dim = 3 * transformer_dim + feature_size;
local span_pair_embedding_dim = 3 * span_embedding_dim + feature_size;

local local_train_path = "./all_chapter_2.jsonl";
local local_val_path = "./2375715.jsonl";
local local_test_path = "./2375715.jsonl";


{
  "evaluate_on_test": true,
  "dataset_reader": {
    "type": 'stary',
    "token_indexers": {
      "tokens": {
        "type": 'pretrained_transformer_mismatched',
        "model_name": transformer_model,
        "max_length": max_length,
      },
    },
    "max_span_width": max_span_width,
    "max_sentences": 1,
  },
  "validation_dataset_reader": {
    "type": 'staryEval',
    "token_indexers": {
      "tokens": {
        "type": 'pretrained_transformer_mismatched',
        "model_name": transformer_model,
        "max_length": max_length,
      },
    },
    "max_span_width": max_span_width,
  },
  "train_data_path": local_train_path,
  "validation_data_path": local_val_path,
  "test_data_path": local_test_path,
  "model": {
    "type": 'from_archive',
    "archive_file": "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz",
  },
  /***
  "model": {
    "type": 'coref',
    "text_field_embedder": {
    "token_embedders": {
        "tokens": {
            "type": 'pretrained_transformer_mismatched',
            "model_name": transformer_model,
            "max_length": max_length,
        },
      },
    },
    "context_layer": {
        "type": 'pass_through',
        "input_dim": transformer_dim,
    },
    "mention_feedforward": {
        "input_dim": span_embedding_dim,
        "num_layers": 2,
        "hidden_dims": 1500,
        "activations": 'relu',
        "dropout": 0.3,
    },
    "antecedent_feedforward": {
        "input_dim": span_pair_embedding_dim,
        "num_layers": 2,
        "hidden_dims": 1500,
        "activations": 'relu',
        "dropout": 0.3,
    },
    "initializer": {
      "regexes": [
        ['.*_span_updating_gated_sum.*weight', {"type": 'xavier_normal' }],
        ['.*linear_layers.*weight', {"type": 'xavier_normal' }],
        ['.*scorer.*weight', {"type": 'xavier_normal' }],
        ['_distance_embedding.weight', {"type": 'xavier_normal' }],
        ['_span_width_embedding.weight', {"type": 'xavier_normal' }],
        ['_context_layer._module.weight_ih.*', {"type": 'xavier_normal' }],
        ['_context_layer._module.weight_hh.*', {"type": 'orthogonal' }]
      ],
    },
    "feature_size": feature_size,
    "max_span_width": max_span_width,
    "spans_per_word": 0.4,
    "max_antecedents": 50,
    "coarse_to_fine": true,
    "inference_order": 2,
  },
  ***/
  "data_loader": {
    "batch_sampler": {
      "type": 'bucket',
      # Explicitly specifying sorting keys since the guessing heuristic could get it wrong
      # as we a span field.
      "sorting_keys": ['text'],
      "batch_size": 1,
    },
  },

  "trainer": {
    "num_epochs": 10,
    "patience" : 5,
    "validation_metric": '+coref_f1',
    "learning_rate_scheduler": {
      "type": 'slanted_triangular',
      "cut_frac": 0.06,
    },
    "optimizer": {
      "type": 'huggingface_adamw',
      "lr": 3e-4,
      "parameter_groups": [
        [['.*transformer.*'], {"lr": 1e-5}]
      ],
    },
  },
  #"distributed": { "cuda_devices": [0, 1],
  #"ddp_accelerator": {
  #    "type": "torch",
  #    "find_unused_parameters": true
  #  }
  #}
}

