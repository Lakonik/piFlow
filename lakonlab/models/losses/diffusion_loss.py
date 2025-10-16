import math
import torch
import torch.distributed as dist

from functools import partial
from mmgen.models import MODULES
from mmgen.models.losses.ddpm_loss import DDPMLoss, mse_loss, reduce_loss
from mmgen.models.losses.utils import weighted_loss

from lakonlab.ops.gmflow_ops.gmflow_ops import gm_logprob


@weighted_loss
def gaussian_nll_loss(pred, target, logstd, eps=1e-4):
    inverse_std = torch.exp(-logstd).clamp(max=1 / eps)
    diff_weighted = (pred - target) * inverse_std
    loss = 0.5 * (diff_weighted.square() + math.log(2 * math.pi)) + logstd
    return loss


@weighted_loss
def gaussian_mixture_nll_loss(
        pred_means, target, pred_logstds, pred_logweights):
    """
    Args:
        pred_means (torch.Tensor): Shape (bs, *, num_gaussians, c, h, w)
        target (torch.Tensor): Shape (bs, *, c, h, w)
        pred_logstds (torch.Tensor): Shape (bs, *, 1 or num_gaussians, 1 or c, 1 or h, 1 or w)
        pred_logweights (torch.Tensor): Shape (bs, *, num_gaussians, 1, h, w)

    Returns:
        torch.Tensor: Shape (bs, *, h, w)
    """
    num_channels = pred_means.size(-3)
    loss = -gm_logprob(
        dict(
            means=pred_means,
            logstds=pred_logstds,
            logweights=pred_logweights),
        target.unsqueeze(-4))[0]
    return loss.squeeze(-3) / num_channels


@MODULES.register_module()
class DiffusionMSELoss(DDPMLoss):
    _default_data_info = dict(pred='eps_t_pred', target='noise')

    def __init__(self,
                 rescale_mode='constant',
                 rescale_cfg=dict(scale=1.0),
                 sampler=None,
                 weight=None,
                 log_cfgs=None,
                 reduction='mean',
                 data_info=None,
                 loss_name='loss_mse'):
        super().__init__(rescale_mode=rescale_mode,
                         rescale_cfg=rescale_cfg,
                         log_cfgs=log_cfgs,
                         weight=weight,
                         sampler=sampler,
                         reduction=reduction,
                         loss_name=loss_name)

        self.data_info = self._default_data_info \
            if data_info is None else data_info

        self.loss_fn = partial(mse_loss, reduction='flatmean')

    def _forward_loss(self, outputs_dict):
        """Forward function for loss calculation.
        Args:
            outputs_dict (dict): Outputs of the model used to calculate losses.

        Returns:
            torch.Tensor: Calculated loss.
        """
        loss_input_dict = {
            k: outputs_dict[v]
            for k, v in self.data_info.items()
        }
        loss = self.loss_fn(**loss_input_dict) * 0.5
        return loss


@MODULES.register_module()
class DiffusionNLLLoss(DDPMLoss):
    _default_data_info = dict(pred='u_t_pred', target='u_t', logstd='logstd')

    def __init__(self,
                 rescale_mode='constant',
                 rescale_cfg=dict(scale=1.0),
                 log_cfgs=None,
                 data_info=None,
                 reduction='mean',
                 loss_name='loss_nll'):
        super().__init__(
            rescale_mode=rescale_mode,
            rescale_cfg=rescale_cfg,
            log_cfgs=log_cfgs,
            reduction=reduction,
            loss_name=loss_name)
        self.data_info = self._default_data_info \
            if data_info is None else data_info
        self.loss_fn = partial(gaussian_nll_loss, reduction='flatmean')
        if log_cfgs is not None and log_cfgs.get('type', None) == 'quartile':
            for i in range(4):
                self.register_buffer(f'loss_quartile_{i}', torch.zeros((1, ), dtype=torch.float))
                self.register_buffer(f'var_quartile_{i}', torch.ones((1, ), dtype=torch.float))
                self.register_buffer(f'count_quartile_{i}', torch.zeros((1, ), dtype=torch.long))

    @torch.no_grad()
    def collect_log(self, loss, var, timesteps):
        if not self.log_fn_list:
            return

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder_l = [torch.zeros_like(loss) for _ in range(ws)]
            placeholder_v = [torch.zeros_like(var) for _ in range(ws)]
            placeholder_t = [torch.zeros_like(timesteps) for _ in range(ws)]
            dist.all_gather(placeholder_l, loss)
            dist.all_gather(placeholder_v, var)
            dist.all_gather(placeholder_t, timesteps)
            loss = torch.cat(placeholder_l, dim=0)
            var = torch.cat(placeholder_v, dim=0)
            timesteps = torch.cat(placeholder_t, dim=0)
        log_vars = dict()

        if (dist.is_initialized()
                and dist.get_rank() == 0) or not dist.is_initialized():
            for log_fn in self.log_fn_list:
                log_vars.update(log_fn(loss, var, timesteps))
        self.log_vars = log_vars

    @torch.no_grad()
    def quartile_log_collect(self,
                             loss,
                             var,
                             timesteps,
                             total_timesteps,
                             prefix_name,
                             reduction='mean',
                             momentum=0.1):
        quartile = (timesteps / total_timesteps * 4)
        quartile = quartile.to(torch.long).clamp(min=0, max=3)

        log_vars = dict()

        for idx in range(4):
            quartile_mask = quartile == idx
            quartile_count = torch.count_nonzero(quartile_mask).reshape(1)
            if quartile_count > 0:
                loss_quartile = reduce_loss(loss[quartile_mask], reduction).reshape(1)
                var_quartile = reduce_loss(var[quartile_mask], reduction).reshape(1)

                cur_weight = 1 - torch.exp(-momentum * quartile_count)
                getattr(self, f'count_quartile_{idx}').add_(quartile_count)
                total_weight = 1 - torch.exp(-momentum * getattr(self, f'count_quartile_{idx}'))
                cur_weight /= total_weight.clamp(min=1e-4)
                getattr(self, f'loss_quartile_{idx}').mul_(1 - cur_weight).add_(loss_quartile * cur_weight)
                getattr(self, f'var_quartile_{idx}').mul_(1 - cur_weight).add_(var_quartile * cur_weight)

            log_vars[f'{prefix_name}_quartile_{idx}'] = getattr(self, f'loss_quartile_{idx}').item()
            log_vars[f'{prefix_name}_var_quartile_{idx}'] = getattr(self, f'var_quartile_{idx}').item()

        return log_vars

    def _forward_loss(self, outputs_dict):
        loss_input_dict = {
            k: outputs_dict[v]
            for k, v in self.data_info.items()
        }
        loss = self.loss_fn(**loss_input_dict)
        return loss

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            assert isinstance(args[0], dict), (
                'You should offer a dictionary containing network outputs '
                'for building up computational graph of this loss module.')
            output_dict = args[0]
        elif 'output_dict' in kwargs:
            assert len(args) == 0, (
                'If the outputs dict is given in keyworded arguments, no'
                ' further non-keyworded arguments should be offered.')
            output_dict = kwargs.pop('outputs_dict')
        else:
            raise NotImplementedError(
                'Cannot parsing your arguments passed to this loss module.'
                ' Please check the usage of this module')

        # check keys in output_dict
        assert 'timesteps' in output_dict, (
            '\'timesteps\' is must for DDPM-based losses, but found'
            f'{output_dict.keys()} in \'output_dict\'')

        timesteps = output_dict['timesteps']
        loss = self._forward_loss(output_dict)

        with torch.no_grad():
            var = torch.exp(output_dict['logstd'] * 2)  # (bs, *)
            if 'weight' in self.data_info:
                weight = output_dict[self.data_info['weight']]  # (bs, *)
                weight_norm_factor = weight.flatten(1).mean(dim=1).clamp(min=1e-6)
                _var = (var * weight).flatten(1).mean(dim=1) / weight_norm_factor
                _loss = loss / weight_norm_factor
            else:
                _var = var.flatten(1).mean(dim=1)
                _loss = loss

            # update log_vars of this class
            self.collect_log(_loss, _var, timesteps=timesteps)  # Mod: log after rescaling

        loss_rescaled = self.rescale_fn(loss, timesteps)
        return reduce_loss(loss_rescaled, self.reduction)


@MODULES.register_module()
class GMFlowNLLLoss(DiffusionNLLLoss):
    _default_data_info = dict(
        pred_means='means',
        target='u_t',
        pred_logstds='logstds',
        pred_logweights='logweights')

    def __init__(self,
                 rescale_mode='constant',
                 rescale_cfg=dict(scale=1.0),
                 log_cfgs=None,
                 data_info=None,
                 reduction='mean',
                 loss_name='loss_nll'):
        super().__init__(
            rescale_mode=rescale_mode,
            rescale_cfg=rescale_cfg,
            log_cfgs=log_cfgs,
            reduction=reduction,
            loss_name=loss_name)
        self.data_info = self._default_data_info \
            if data_info is None else data_info
        self.loss_fn = partial(gaussian_mixture_nll_loss, reduction='flatmean')
        if log_cfgs is not None and log_cfgs.get('type', None) == 'quartile':
            for i in range(4):
                self.register_buffer(f'loss_quartile_{i}', torch.zeros((1,), dtype=torch.float))
                self.register_buffer(f'var_quartile_{i}', torch.ones((1,), dtype=torch.float))
                self.register_buffer(f'count_quartile_{i}', torch.zeros((1,), dtype=torch.long))

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            assert isinstance(args[0], dict), (
                'You should offer a dictionary containing network outputs '
                'for building up computational graph of this loss module.')
            output_dict = args[0]
        elif 'output_dict' in kwargs:
            assert len(args) == 0, (
                'If the outputs dict is given in keyworded arguments, no'
                ' further non-keyworded arguments should be offered.')
            output_dict = kwargs.pop('outputs_dict')
        else:
            raise NotImplementedError(
                'Cannot parsing your arguments passed to this loss module.'
                ' Please check the usage of this module')

        # check keys in output_dict
        assert 'timesteps' in output_dict, (
            '\'timesteps\' is must for DDPM-based losses, but found'
            f'{output_dict.keys()} in \'output_dict\'')

        timesteps = output_dict['timesteps']
        loss = self._forward_loss(output_dict)

        with torch.no_grad():
            weights = output_dict['logweights'].exp()
            mean = (weights * output_dict['means']).sum(-4, keepdim=True)  # (bs, *, 1, c, h, w)
            var = (weights * ((output_dict['means'] - mean).square()
                              + (output_dict['logstds'] * 2).exp())).sum(-4)  # (bs, *, c, h, w)
            if 'weight' in self.data_info:
                weight = output_dict[self.data_info['weight']].unsqueeze(-3)  # (bs, *, 1, h, w)
                weight_norm_factor = weight.flatten(1).mean(dim=1).clamp(min=1e-6)
                _var = (var * weight).flatten(1).mean(dim=1) / weight_norm_factor
                _loss = loss / weight_norm_factor
            else:
                _var = var.flatten(1).mean(dim=1)
                _loss = loss

            # update log_vars of this class
            self.collect_log(_loss, _var, timesteps=timesteps)  # Mod: log after rescaling

        loss_rescaled = self.rescale_fn(loss, timesteps)
        return reduce_loss(loss_rescaled, self.reduction)
