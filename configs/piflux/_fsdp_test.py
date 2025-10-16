module_wrapper = 'fsdp'
fsdp_kwargs = dict(
    wrap_frozen_modules=True,  # shard all modules
    ignore_frozen_parameters=False,  # shard all parameters
    fsdp_modules=['diffusers.models.transformers.transformer_flux.FluxTransformerBlock',
                  'diffusers.models.transformers.transformer_flux.FluxSingleTransformerBlock'],
    exclude_keys=['vae'],
)
