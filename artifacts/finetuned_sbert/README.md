---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:35766
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: 'Pendidikan Master''s pengalaman 4 tahun. Skill teknis: [''vendor
    management'', ''strategy'', ''research'', ''education'', ''c'', ''google analytics'',
    ''operations'', ''compliance'', ''scrum'', ''kanban'', ''project management'',
    ''financial analysis'', ''excel'', ''sql'']'
  sentences:
  - 'Posisi Channel Management Consultant pendidikan minimum Master''s. Dibutuhkan
    skill teknis: [''strategy'', ''project management'']'
  - 'Soft skill yang dibutuhkan: []'
  - 'Soft skill yang dibutuhkan: [''leadership'', ''communication'', ''problem solving'']'
- source_sentence: 'Soft skill kandidat: []'
  sentences:
  - 'Posisi Accounting Clerk pendidikan minimum Master''s. Dibutuhkan skill teknis:
    [''attention to detail'', ''education'', ''accounting'']'
  - 'Soft skill yang dibutuhkan: [''communication'']'
  - 'Soft skill yang dibutuhkan: [''communication'', ''adaptability'', ''teamwork'',
    ''problem solving'']'
- source_sentence: 'Soft skill kandidat: [''teamwork'', ''communication'', ''time
    management'', ''analytical'']'
  sentences:
  - 'Posisi Physician pendidikan minimum PhD. Dibutuhkan skill teknis: [''r'', ''education'']'
  - 'Soft skill yang dibutuhkan: []'
  - 'Posisi National Sales Manager pendidikan minimum Bachelor''s. Dibutuhkan skill
    teknis: [''inventory management'']'
- source_sentence: 'Pendidikan Master''s pengalaman 10 tahun. Skill teknis: [''vendor
    management'', ''strategy'', ''payroll'', ''risk management'', ''education'', ''operations'',
    ''safety'', ''sap'', ''compliance'', ''accounting'', ''statistics'', ''hipaa'',
    ''employee relations'', ''hris'']'
  sentences:
  - 'Posisi Pega Systems Architect pendidikan minimum Master''s. Dibutuhkan skill
    teknis: [''ci/cd'', ''customer service'', ''scrum'', ''crm'', ''devops'']'
  - 'Posisi Content Assistant pendidikan minimum Bachelor''s. Dibutuhkan skill teknis:
    [''copywriting'', ''troubleshooting'']'
  - 'Posisi Ugc Content Creator pendidikan minimum Bachelor''s. Dibutuhkan skill teknis:
    [''attention to detail'', ''strategy'', ''multitasking'', ''social media'']'
- source_sentence: 'Soft skill kandidat: [''communication'', ''leadership'']'
  sentences:
  - 'Posisi Nurse Practitioner Nurse Practitioner pendidikan minimum Bachelor''s.
    Dibutuhkan skill teknis: [''education'', ''c'', ''emr'', ''patient care'']'
  - 'Soft skill yang dibutuhkan: [''communication'']'
  - 'Soft skill yang dibutuhkan: [''leadership'', ''communication'', ''negotiation'']'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'mean', 'include_prompt': True})
  (2): Normalize({})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    "Soft skill kandidat: ['communication', 'leadership']",
    "Soft skill yang dibutuhkan: ['leadership', 'communication', 'negotiation']",
    "Posisi Nurse Practitioner Nurse Practitioner pendidikan minimum Bachelor's. Dibutuhkan skill teknis: ['education', 'c', 'emr', 'patient care']",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.2415, 0.0621],
#         [0.2415, 1.0000, 0.0309],
#         [0.0621, 0.0309, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 35,766 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                          | sentence_1                                                                          | label                                                          |
  |:---------|:------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                              | string                                                                              | float                                                          |
  | modality | text                                                                                | text                                                                                |                                                                |
  | details  | <ul><li>min: 11 tokens</li><li>mean: 35.92 tokens</li><li>max: 101 tokens</li></ul> | <ul><li>min: 13 tokens</li><li>mean: 32.45 tokens</li><li>max: 101 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.24</li><li>max: 0.9</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                        | sentence_1                                                                                                                                                                     | label               |
  |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|
  | <code>Pendidikan Master's pengalaman 3 tahun. Skill teknis: ['research', 'education', 'customer service', 'supply chain', 'logistics', 'operations', 'procurement', 'teaching', 'sap', 'compliance', 'accounting', 'r', 'mentoring']</code>                       | <code>Posisi Computer Aided Design Designer pendidikan minimum Bachelor's. Dibutuhkan skill teknis: ['attention to detail', 'typography', 'adobe xd', 'graphic design']</code> | <code>0.0096</code> |
  | <code>Pendidikan Master's pengalaman 4 tahun. Skill teknis: ['vendor management', 'strategy', 'research', 'education', 'c', 'google analytics', 'operations', 'compliance', 'scrum', 'kanban', 'project management', 'financial analysis', 'excel', 'sql']</code> | <code>Posisi Channel Management Consultant pendidikan minimum Master's. Dibutuhkan skill teknis: ['strategy', 'project management']</code>                                     | <code>0.6568</code> |
  | <code>Soft skill kandidat: ['leadership']</code>                                                                                                                                                                                                                  | <code>Soft skill yang dibutuhkan: ['communication']</code>                                                                                                                     | <code>0.1717</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss",
      "cos_score_transformation": "torch.nn.modules.linear.Identity"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `do_predict`: False
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 3
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: None
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `enable_jit_checkpoint`: False
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `use_cpu`: False
- `seed`: 42
- `data_seed`: None
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: -1
- `ddp_backend`: None
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `auto_find_batch_size`: False
- `full_determinism`: False
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `use_cache`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.4472 | 500  | 0.0583        |
| 0.8945 | 1000 | 0.0431        |
| 1.3417 | 1500 | 0.0413        |
| 1.7889 | 2000 | 0.0401        |
| 2.2361 | 2500 | 0.0399        |
| 2.6834 | 3000 | 0.0383        |


### Training Time
- **Training**: 10.3 minutes

### Framework Versions
- Python: 3.12.13
- Sentence Transformers: 5.5.1
- Transformers: 5.0.0
- PyTorch: 2.11.0+cu128
- Accelerate: 1.13.0
- Datasets: 4.0.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->