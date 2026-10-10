export const voxAdvancedParameters = new Set([
  'python_path', 'trainer_path',
  'num_workers', 'preprocessing_num_workers', 'max_batch_tokens', 'weight_decay', 'max_grad_norm',
  'max_steps', 'loss_diff_weight', 'loss_stop_weight', 'log_interval', 'lora_dropout',
  'lora_enable_lm', 'lora_enable_dit', 'lora_enable_proj',
]);
export const gptSovitsAdvancedParameters = new Set([
  'python_path', 'trainer_path',
  'gpt.seed', 'gpt.num_workers', 'gpt.initial_learning_rate', 'gpt.final_learning_rate',
  'gpt.decay_steps', 'gpt.dpo', 'gpt.save_latest', 'sovits.seed', 'sovits.adam_beta1',
  'sovits.adam_beta2', 'sovits.adam_epsilon', 'sovits.lr_decay', 'sovits.log_interval', 'sovits.save_latest',
]);
