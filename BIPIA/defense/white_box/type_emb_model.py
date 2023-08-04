import torch
from transformers import LlamaForCausalLM, LlamaConfig

class LlamaModelWTypeEmbedCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        config.type_vocab_size = 2
        self.type_embeddings = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.type_embeddings.weight.data.zero_()

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        type_ids = None,
        past_key_values = None,
        inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        batch_size, seq_len, hidden_size = inputs_embeds.size()

        if type_ids is None:
            type_ids = torch.zeros(batch_size, seq_len, device=inputs_embeds.device)

        type_embeds = self.type_embeddings(type_ids.long())
        inputs_embeds = inputs_embeds + type_embeds.to(inputs_embeds.device)

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def prepare_inputs_for_generation(
        self, input_ids, type_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if type_ids is not None:
            if past_key_values:
                type_ids = type_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {   
                "type_ids": type_ids,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs