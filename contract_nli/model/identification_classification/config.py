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

def update_config(config, *, impossible_strategy, class_loss_weight, pre_seq_len, max_query_length, template):
    class IdentificationClassificationConfig(type(config)):
        def __init__(self, impossible_strategy='ignore',
                     class_loss_weight=1.0, pre_seq_len=100, template=1, **kwargs):
            super().__init__(**kwargs)
            self.impossible_strategy = impossible_strategy
            self.class_loss_weight = class_loss_weight
            self.pre_seq_len = pre_seq_len
            self.template = template

        @classmethod
        def from_config(cls, config, *, impossible_strategy,
                        class_loss_weight, pre_seq_len, max_query_length, template):
            kwargs = config.to_dict()
            assert 'impossible_strategy' not in kwargs
            kwargs['impossible_strategy'] = impossible_strategy
            assert 'class_loss_weight' not in kwargs
            kwargs['class_loss_weight'] = class_loss_weight
            assert 'pre_seq_len' not in kwargs
            kwargs['pre_seq_len'] = pre_seq_len
            assert 'max_query_length' not in kwargs
            kwargs['max_query_length'] = max_query_length
            assert 'template' not in kwargs
            kwargs['template'] = template

            return cls(**kwargs)

    return IdentificationClassificationConfig.from_config(
        config, impossible_strategy=impossible_strategy,
        class_loss_weight=class_loss_weight, pre_seq_len=pre_seq_len,
        max_query_length=max_query_length, template=template)
