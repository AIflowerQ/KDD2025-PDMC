import json

import numpy as np
import torch

from data.catalog.datasat_catalog import BnPlusCatalog
from evaluation.evaluator.evaluator import Evaluator
from evaluation.prepare_metrics_para import prepare_metrics_para_for_bn_plus
from evaluation.utils import nni_target_v3
from predict_models.api.predict_models import NonLinearModel
from recourse_methods.autoencoder.autoencoder import ConditionalVariationalAutoencoder
from recourse_methods.autoencoder.autoencoder_prepare import prepare_autoencoder_xy_train, \
    prepare_autoencoder_delta_x_given_x_train
from recourse_methods.catalog.OURS.model_v3 import OURS_V3
from utils.utils import set_random_seed, merge_dict, two_level_mean_merge_dict_func

# torch.cuda.set_device(6)
DEVICE: torch.device = torch.device('cuda')


DATA_PATH: str = 'CHOOSE YOUR DATA'
data_manager: BnPlusCatalog = BnPlusCatalog(DATA_PATH)
assert data_manager.is_transformed

p_x_given_y_cvae_config: dict = {
    'condition_dim': 2,
    'mutable_mask': np.ones(len(data_manager.feature_columns_order)).astype(bool)
}

p_x_given_y_cvae_train_config: dict = {}


p_delta_x_given_x_cvae_config: dict = {
    'condition_dim': len(data_manager.feature_columns_order),
    'mutable_mask': np.ones(len(data_manager.feature_columns_order)).astype(bool)
}

p_delta_x_given_x_cvae_train_config: dict = {}

cf_para: dict = {
        "max_iter": 150,
        "log_interval": 10,
        "silent": True
 }


RANDOM_SEED_LIST: list = [17373331, 17373423]
MODEL_PATH: str = 'CHOOSE YOUR ML MODEL'
HYPER_PARA_PATH: str = 'CHOOSE YOUR HYPER PARA'


def main(random_seed: int):
    set_random_seed(random_seed)
    p_x_given_y_cvae: ConditionalVariationalAutoencoder = \
        ConditionalVariationalAutoencoder(data_manger=data_manager, **p_x_given_y_cvae_config).to(DEVICE)
    prepare_autoencoder_xy_train(
        ae=p_x_given_y_cvae, data_manager=data_manager,
        device=DEVICE, training_para=p_x_given_y_cvae_train_config
    )
    p_delta_x_given_x_cvae: ConditionalVariationalAutoencoder = \
        ConditionalVariationalAutoencoder(data_manger=data_manager, **p_delta_x_given_x_cvae_config).to(DEVICE)
    prepare_autoencoder_delta_x_given_x_train(
        ae=p_delta_x_given_x_cvae, data_manager=data_manager,
        device=DEVICE, training_para=p_delta_x_given_x_cvae_train_config
    )
    nonlinear_model: NonLinearModel = torch.load(MODEL_PATH, map_location='cpu')
    nonlinear_model.eval()
    nonlinear_model.to(DEVICE)
    ours: OURS_V3 = OURS_V3(
        ml_model=nonlinear_model, data_manager=data_manager,
        p_x_given_y_cvae=p_x_given_y_cvae,
        p_delta_x_given_x_cvae=p_delta_x_given_x_cvae,
        hyperparams=cf_para
    )
    metrics_para, qualified_metric_dict = prepare_metrics_para_for_bn_plus(data_manager)
    evaluator: Evaluator = \
        Evaluator(data_manager, nonlinear_model, ours, metrics_para, qualified_metric_dict, 'inverse', DEVICE)
    evaluate_result: dict = evaluator.evaluate()
    print(json.dumps(evaluate_result, indent=4))

    return evaluate_result


if __name__ == '__main__':
    params: dict = json.load(open(HYPER_PARA_PATH, 'r'))

    p_x_given_y_cvae_config['layers'] = \
        [len(data_manager.feature_columns_order)] + (int(params['x_y_deep']) * [int(params['x_y_wide'])])
    p_x_given_y_cvae_train_config['batch_size'] = int(params['x_y_batch_size'])
    p_x_given_y_cvae_train_config['lr'] = params['x_y_lr']
    p_x_given_y_cvae_train_config['lambda_reg'] = params['x_y_lambda_reg']
    p_x_given_y_cvae_train_config['epochs'] = int(params['x_y_epochs'])
    p_x_given_y_cvae_train_config['kl_weight'] = params['x_y_kl_weight']


    p_delta_x_given_x_cvae_config['layers'] = \
        [len(data_manager.feature_columns_order)] + (int(params['x_x_deep']) * [int(params['x_x_wide'])])
    p_delta_x_given_x_cvae_train_config['batch_size'] = int(params['x_x_batch_size'])
    p_delta_x_given_x_cvae_train_config['lr'] = params['x_x_lr']
    p_delta_x_given_x_cvae_train_config['lambda_reg'] = params['x_x_lambda_reg']
    p_delta_x_given_x_cvae_train_config['epochs'] = int(params['x_x_epochs'])
    p_delta_x_given_x_cvae_train_config['kl_weight'] = params['x_x_kl_weight']

    cf_para['md_epsilon'] = params['cf_md_epsilon']
    cf_para['l2_coe'] = params['cf_l2_coe']
    cf_para['delta_x_l2_coe'] = params['cf_delta_x_l2_coe']
    cf_para['delta_x_given_x_l2_coe'] = params['cf_delta_x_given_x_l2_coe']
    cf_para['delta_x_md_coe'] = params['cf_delta_x_md_coe']
    cf_para['delta_x_given_x_md_coe'] = params['cf_delta_x_given_x_md_coe']
    cf_para['x_md_coe'] = params['cf_x_md_coe']
    cf_para['lr'] = params['cf_lr']
    cf_para['samples_num'] = int(params['cf_samples_num'])

    print('p_x_given_y_config:')
    print(p_x_given_y_cvae_config)
    print(json.dumps(p_x_given_y_cvae_train_config, indent=4))

    print('p_delta_x_given_x_config:')
    print(p_delta_x_given_x_cvae_config)
    print(json.dumps(p_delta_x_given_x_cvae_train_config, indent=4))

    print('cf_para:')
    print(json.dumps(cf_para, indent=4))

    results_list: list = list()
    for seed in RANDOM_SEED_LIST:
        print(f'Begin Seed {seed}')
        results_list.append(main(seed))
        print(f'End Seed {seed}')

    merged_result: dict = merge_dict(results_list, two_level_mean_merge_dict_func)
    print('Final Results')
    print(json.dumps(merged_result, indent=4))

    delta_x_kde_percentile: float = merged_result["Delta X Manifold"]['kde_percentile']
    x_kde_percentile: float = merged_result["X Manifold"]['kde_percentile']
    qualified_rate: float = merged_result['qualified_rate']

    valid_rate: float = merged_result['valid_rate']

    report: dict = {
        'default': nni_target_v3(qualified_rate, delta_x_kde_percentile, x_kde_percentile, valid_rate),
        'delta_x_kde_percentile': delta_x_kde_percentile,
        'x_kde_percentile': x_kde_percentile,
        'qualified_rate': qualified_rate,
        'valid_rate': valid_rate
    }

    print('Report:')
    print(json.dumps(report, indent=4))


