# Copyright (c) 2021, Hitachi America Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from transformers.models.bert import BertPreTrainedModel, BertModel, BertConfig
from transformers.models.gpt2 import GPT2Model, GPT2Config
from transformers.utils import logging

from contract_nli.dataset.loader import NLILabel
from contract_nli.model.identification_classification.model_output import \
    IdentificationClassificationModelOutput

logger = logging.get_logger(__name__)


class BertForIdentificationClassification(BertPreTrainedModel):

    IMPOSSIBLE_STRATEGIES = {'ignore', 'label', 'not_mentioned'}

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=True)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.class_outputs = nn.Linear(config.hidden_size, 3)
        self.span_outputs = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.model_type: str = config.model_type
        self.template = config.template

        if config.impossible_strategy not in self.IMPOSSIBLE_STRATEGIES:
            raise ValueError(
                f'impossible_strategy must be one of {self.IMPOSSIBLE_STRATEGIES}')
        self.impossible_strategy = config.impossible_strategy

        self.class_loss_weight = config.class_loss_weight
        self.init_weights()

        # ------------------------------------------

        self.pre_seq_len = config.pre_seq_len
        self.max_query_length = config.max_query_length
        self.embeddings = self.bert.embeddings

        self.context_encoder_config = BertConfig.from_pretrained('bert-base-uncased')
        self.context_encoder = BertModel(self.context_encoder_config)
        for param in self.context_encoder.parameters():
            param.requires_grad = False

        self.prompt_encoder_config = BertConfig.from_pretrained('bert-base-uncased')
        self.prompt_encoder_config.num_hidden_layers = 2
        self.prompt_encoder = BertModel(self.prompt_encoder_config)

        self.mid_dim = config.mid_dim if hasattr(config, 'mid_dim') else 512
        self.control_trans = torch.nn.Sequential(
            torch.nn.Linear(self.context_encoder_config.hidden_size, self.mid_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(self.mid_dim, self.prompt_encoder_config.n_layer * 2 *
                            self.prompt_encoder_config.n_embd *
                            self.prompt_encoder_config.n_head)
        )

        self.prompt_tokens = torch.arange(1, self.pre_seq_len + 1).long()
        self.transform = torch.nn.Linear(config.hidden_size, 1)
        self.prompt_embed = torch.nn.Embedding(self.pre_seq_len, config.hidden_size) # size = (?,?)

    def get_context(self, input_ids, position_ids, attention_mask):
        # get hypothesis statements as contexts
        query_ids = input_ids[:, self.max_query_length]
        if position_ids is not None:
            position_ids = position_ids[:, self.max_query_length]
        attention_mask = attention_mask[:, self.max_query_length]

        # get context embeddings and convert them to past key values
        self.context_encoder.eval()
        with torch.no_grad():
            context = self.context_encoder(
                input_ids=query_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            ).last_hidden_state  # (bs, seq_len, hidden_size)
        past_key_values = self.control_trans(context)

        # adjust past key values dimensions to fit into the prompt encoder
        bsz, seqlen, _ = past_key_values.shape
        config = self.prompt_encoder_config
        past_key_values = past_key_values.view(bsz, seqlen, config.n_layer * 2, config.n_head,
                                               config.n_embd)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        # past_key_values is of shape (12, 2, 8, 12, 256, 768) -> (n_layers, 2, bsz, n_head, seqlen, hidden_size)

        return past_key_values

    def get_prompt(self, past_key_values, batch_size):
        prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        prompts = self.prompt_encoder(input_ids=prompt_tokens, past_key_values=past_key_values)
        # output size = (bsz, seqlen, hidden_size)
        prompts = self.transform(prompts)  # size has to become (bsz, seqlen, 1)
        prompt_min = torch.min(prompts)
        prompt_max = torch.max(prompts)
        prompts = (prompts+prompt_min) / (prompt_max-prompt_min) * self.pre_seq_len
        # have to be in the range 0~pre_seq_len
        prompts = prompts.to(torch.long)
        prompts = self.prompt_embed(prompts)
        return prompts

    # ------------------------------------------

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        class_labels=None,
        span_labels=None,
        p_mask=None,
        valid_span_missing_in_context=None,
    ) -> IdentificationClassificationModelOutput:

        # ------------------------------------------

        batch_size = input_ids.shape[0]
        input_ids = torch.tensor(input_ids).to(self.bert.device).long()
        raw_embedding = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )

        context_as_past_key_values = self.get_context(input_ids, position_ids, attention_mask)
        prompts = self.get_prompt(context_as_past_key_values, batch_size)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        prompt_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prompt_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state

        sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        first_token_tensor = sequence_output[:, 0]
        pooled_output = self.bert.pooler.dense(first_token_tensor)
        pooled_output = self.bert.pooler.activation(pooled_output)

        pooled_output = self.dropout(pooled_output)
        logits_cls = self.class_outputs(pooled_output)

        sequence_output = self.dropout(sequence_output)
        logits_span = self.span_outputs(sequence_output)

        # ------------------------------------------

        if class_labels is not None:
            assert p_mask is not None
            assert span_labels is not None
            assert valid_span_missing_in_context is not None

            loss_fct = nn.CrossEntropyLoss()
            if self.impossible_strategy == 'ignore':
                class_labels = torch.where(
                    valid_span_missing_in_context == 0, class_labels,
                    torch.tensor(loss_fct.ignore_index).type_as(class_labels)
                )
            elif self.impossible_strategy == 'not_mentioned':
                class_labels = torch.where(
                    valid_span_missing_in_context == 0, class_labels,
                    NLILabel.NOT_MENTIONED.value
                )
            loss_cls = self.class_loss_weight * loss_fct(logits_cls, class_labels)

            loss_fct = nn.CrossEntropyLoss()
            active_logits = logits_span.view(-1, 2)
            active_labels = torch.where(
                p_mask.view(-1) == 0, span_labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(span_labels)
            )
            loss_span = loss_fct(active_logits, active_labels)
            loss = loss_cls + loss_span
        else:
            loss, loss_cls, loss_span = None, None, None

        return IdentificationClassificationModelOutput(
            loss=loss,
            loss_cls=loss_cls,
            loss_span=loss_span,
            class_logits=logits_cls,
            span_logits=logits_span
        )
