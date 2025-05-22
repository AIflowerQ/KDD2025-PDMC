from causal_structure.causal_model.bn_plus_causal_model import BnPlusStandardProcessCausalModel
from data.catalog.datasat_catalog import BnPlusCatalog
from PDMC_code.evaluation.metrics.causal_constraint import CausalConstraintEvaluation
from PDMC_code.evaluation.metrics.delta_x_manifold import DeltaXManifoldEvaluation
from evaluation.metrics.global_immutable import GlobalImmutableEvaluation
from evaluation.metrics.global_monotonic import GlobalMonotonicEvaluation
from PDMC_code.evaluation.metrics.proximity import ProximityEvaluation
from evaluation.metrics.user_cosine import UserCosineEvaluation
from evaluation.metrics.user_immutable import UserImmutableEvaluation
from evaluation.metrics.x_manifold import XManifoldEvaluation


def prepare_metrics_para_for_bn_plus(data_manager: BnPlusCatalog):

    metrics_para: dict = dict()

    causal_constraint_para: dict = {
        'causal_structure': BnPlusStandardProcessCausalModel(data_manager),
        'thr_list': [0.8, 0.9, 0.95]
    }
    metrics_para[CausalConstraintEvaluation] = causal_constraint_para


    delta_x_manifold_para: dict = {
        'md_epsilon': 1e-3,
        'kde_bw': 0.2
        # 'kde_bw': 0.15
    }
    metrics_para[DeltaXManifoldEvaluation] = delta_x_manifold_para


    global_immutable_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[GlobalImmutableEvaluation] = global_immutable_para


    global_monotonic_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[GlobalMonotonicEvaluation] = global_monotonic_para


    proximity_para: dict = {}
    metrics_para[ProximityEvaluation] = proximity_para


    user_cosine_para: dict = {'eps_thr_list': [0.0, 0.2, 0.3, 0.5, 0.7]}
    metrics_para[UserCosineEvaluation] = user_cosine_para

    user_immutable_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[UserImmutableEvaluation] = user_immutable_para

    x_manifold_para: dict = {
        'md_epsilon': 1e-5,
        'kde_bw': 0.35
    }

    x_manifold_para['target_labels'] = None
    metrics_para[XManifoldEvaluation] = x_manifold_para

    qualified_metric_dict: dict = {
        CausalConstraintEvaluation: 'pdf_thr_0.9',
        GlobalImmutableEvaluation: 'global_immutable_0.005',
        GlobalMonotonicEvaluation: 'global_monotonic_0.005',
        UserImmutableEvaluation: 'user_immutable_0.005',
        UserCosineEvaluation: 'user_cos_0.5'
    }

    return metrics_para, qualified_metric_dict

def prepare_metrics_para_for_mimic():

    metrics_para: dict = dict()

    delta_x_manifold_para: dict = {
        'md_epsilon': 1e-3,
        # 'kde_bw': 0.2
        'kde_bw': 0.15
    }
    metrics_para[DeltaXManifoldEvaluation] = delta_x_manifold_para

    global_immutable_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[GlobalImmutableEvaluation] = global_immutable_para

    global_monotonic_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[GlobalMonotonicEvaluation] = global_monotonic_para

    proximity_para: dict = {}
    metrics_para[ProximityEvaluation] = proximity_para

    user_cosine_para: dict = {'eps_thr_list': [0.0, 0.2, 0.3, 0.5, 0.7]}
    metrics_para[UserCosineEvaluation] = user_cosine_para

    x_manifold_para: dict = {
        'md_epsilon': 1e-5,
        'kde_bw': 0.25
    }

    x_manifold_para['target_labels'] = None
    metrics_para[XManifoldEvaluation] = x_manifold_para

    qualified_metric_dict: dict = {
        GlobalImmutableEvaluation: 'global_immutable_0.005',
        GlobalMonotonicEvaluation: 'global_monotonic_0.005',
        UserCosineEvaluation: 'user_cos_0.5'
    }

    return metrics_para, qualified_metric_dict


def prepare_metrics_para_for_bn_plus_v2(data_manager: BnPlusCatalog):

    metrics_para: dict = dict()

    causal_constraint_para: dict = {
        'causal_structure': BnPlusStandardProcessCausalModel(data_manager),
        'thr_list': [0.8, 0.9, 0.95]
    }
    metrics_para[CausalConstraintEvaluation] = causal_constraint_para


    delta_x_manifold_para: dict = {
        'md_epsilon': 1e-3,
        'kde_bw': 0.2
        # 'kde_bw': 0.15
    }
    metrics_para[DeltaXManifoldEvaluation] = delta_x_manifold_para


    global_immutable_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[GlobalImmutableEvaluation] = global_immutable_para


    global_monotonic_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[GlobalMonotonicEvaluation] = global_monotonic_para


    proximity_para: dict = {}
    metrics_para[ProximityEvaluation] = proximity_para


    user_cosine_para: dict = {'eps_thr_list': [0.0, 0.2, 0.3, 0.5, 0.7]}
    metrics_para[UserCosineEvaluation] = user_cosine_para

    user_immutable_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[UserImmutableEvaluation] = user_immutable_para

    x_manifold_para: dict = {
        'md_epsilon': 1e-5,
        'kde_bw': 0.35
    }

    x_manifold_para['target_labels'] = None
    metrics_para[XManifoldEvaluation] = x_manifold_para

    qualified_metric_dict: dict = {
        CausalConstraintEvaluation: 'pdf_thr_0.9',
        GlobalImmutableEvaluation: 'global_immutable_0.005',
        GlobalMonotonicEvaluation: 'global_monotonic_0.005',
        # UserImmutableEvaluation: 'user_immutable_0.005',
        # UserCosineEvaluation: 'user_cos_0.5'
    }

    return metrics_para, qualified_metric_dict



def prepare_metrics_para_for_bn_plus_data1(data_manager: BnPlusCatalog):

    metrics_para: dict = dict()

    causal_constraint_para: dict = {
        'causal_structure': BnPlusStandardProcessCausalModel(
            data_manager, k1=0.005, b1=5, sigma3=0.5
        ),
        'thr_list': [0.8, 0.9, 0.95]
    }
    metrics_para[CausalConstraintEvaluation] = causal_constraint_para


    delta_x_manifold_para: dict = {
        'md_epsilon': 1e-3,
        'kde_bw': 0.2
        # 'kde_bw': 0.15
    }
    metrics_para[DeltaXManifoldEvaluation] = delta_x_manifold_para


    global_immutable_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[GlobalImmutableEvaluation] = global_immutable_para


    global_monotonic_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[GlobalMonotonicEvaluation] = global_monotonic_para


    proximity_para: dict = {}
    metrics_para[ProximityEvaluation] = proximity_para


    user_cosine_para: dict = {'eps_thr_list': [0.0, 0.2, 0.3, 0.5, 0.7]}
    metrics_para[UserCosineEvaluation] = user_cosine_para

    user_immutable_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[UserImmutableEvaluation] = user_immutable_para

    x_manifold_para: dict = {
        'md_epsilon': 1e-5,
        'kde_bw': 0.35
    }

    x_manifold_para['target_labels'] = None
    metrics_para[XManifoldEvaluation] = x_manifold_para

    qualified_metric_dict: dict = {
        CausalConstraintEvaluation: 'pdf_thr_0.9',
        GlobalImmutableEvaluation: 'global_immutable_0.005',
        GlobalMonotonicEvaluation: 'global_monotonic_0.005',
        # UserImmutableEvaluation: 'user_immutable_0.005',
        # UserCosineEvaluation: 'user_cos_0.5'
    }

    return metrics_para, qualified_metric_dict


def prepare_metrics_para_for_bn_plus_data2(data_manager: BnPlusCatalog):

    metrics_para: dict = dict()

    causal_constraint_para: dict = {
        'causal_structure': BnPlusStandardProcessCausalModel(
            data_manager, k1=0.01, b1=4, sigma3=0.5
        ),
        'thr_list': [0.8, 0.9, 0.95]
    }
    metrics_para[CausalConstraintEvaluation] = causal_constraint_para


    delta_x_manifold_para: dict = {
        'md_epsilon': 1e-3,
        'kde_bw': 0.2
        # 'kde_bw': 0.15
    }
    metrics_para[DeltaXManifoldEvaluation] = delta_x_manifold_para


    global_immutable_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[GlobalImmutableEvaluation] = global_immutable_para


    global_monotonic_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[GlobalMonotonicEvaluation] = global_monotonic_para


    proximity_para: dict = {}
    metrics_para[ProximityEvaluation] = proximity_para


    user_cosine_para: dict = {'eps_thr_list': [0.0, 0.2, 0.3, 0.5, 0.7]}
    metrics_para[UserCosineEvaluation] = user_cosine_para

    user_immutable_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[UserImmutableEvaluation] = user_immutable_para

    x_manifold_para: dict = {
        'md_epsilon': 1e-5,
        'kde_bw': 0.35
    }

    x_manifold_para['target_labels'] = None
    metrics_para[XManifoldEvaluation] = x_manifold_para

    qualified_metric_dict: dict = {
        CausalConstraintEvaluation: 'pdf_thr_0.9',
        GlobalImmutableEvaluation: 'global_immutable_0.005',
        GlobalMonotonicEvaluation: 'global_monotonic_0.005',
        # UserImmutableEvaluation: 'user_immutable_0.005',
        # UserCosineEvaluation: 'user_cos_0.5'
    }

    return metrics_para, qualified_metric_dict


def prepare_metrics_para_for_bn_plus_data3(data_manager: BnPlusCatalog):

    metrics_para: dict = dict()

    causal_constraint_para: dict = {
        'causal_structure': BnPlusStandardProcessCausalModel(
            data_manager, k1=0.001, b1=12, sigma3=0.5
        ),
        'thr_list': [0.8, 0.9, 0.95]
    }
    metrics_para[CausalConstraintEvaluation] = causal_constraint_para


    delta_x_manifold_para: dict = {
        'md_epsilon': 1e-3,
        'kde_bw': 0.2
        # 'kde_bw': 0.15
    }
    metrics_para[DeltaXManifoldEvaluation] = delta_x_manifold_para


    global_immutable_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[GlobalImmutableEvaluation] = global_immutable_para


    global_monotonic_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[GlobalMonotonicEvaluation] = global_monotonic_para


    proximity_para: dict = {}
    metrics_para[ProximityEvaluation] = proximity_para


    user_cosine_para: dict = {'eps_thr_list': [0.0, 0.2, 0.3, 0.5, 0.7]}
    metrics_para[UserCosineEvaluation] = user_cosine_para

    user_immutable_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[UserImmutableEvaluation] = user_immutable_para

    x_manifold_para: dict = {
        'md_epsilon': 1e-5,
        'kde_bw': 0.35
    }

    x_manifold_para['target_labels'] = None
    metrics_para[XManifoldEvaluation] = x_manifold_para

    qualified_metric_dict: dict = {
        CausalConstraintEvaluation: 'pdf_thr_0.9',
        GlobalImmutableEvaluation: 'global_immutable_0.005',
        GlobalMonotonicEvaluation: 'global_monotonic_0.005',
        # UserImmutableEvaluation: 'user_immutable_0.005',
        # UserCosineEvaluation: 'user_cos_0.5'
    }

    return metrics_para, qualified_metric_dict


def prepare_metrics_para_for_bn_plus_data4(data_manager: BnPlusCatalog):

    metrics_para: dict = dict()

    causal_constraint_para: dict = {
        'causal_structure': BnPlusStandardProcessCausalModel(
            data_manager, k1=0.05, b1=3, sigma3=0.5
        ),
        'thr_list': [0.8, 0.9, 0.95]
    }
    metrics_para[CausalConstraintEvaluation] = causal_constraint_para


    delta_x_manifold_para: dict = {
        'md_epsilon': 1e-3,
        'kde_bw': 0.2
        # 'kde_bw': 0.15
    }
    metrics_para[DeltaXManifoldEvaluation] = delta_x_manifold_para


    global_immutable_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[GlobalImmutableEvaluation] = global_immutable_para


    global_monotonic_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[GlobalMonotonicEvaluation] = global_monotonic_para


    proximity_para: dict = {}
    metrics_para[ProximityEvaluation] = proximity_para


    user_cosine_para: dict = {'eps_thr_list': [0.0, 0.2, 0.3, 0.5, 0.7]}
    metrics_para[UserCosineEvaluation] = user_cosine_para

    user_immutable_para: dict = {'eps_thr_list': [3e-2, 1e-2, 5e-3]}
    metrics_para[UserImmutableEvaluation] = user_immutable_para

    x_manifold_para: dict = {
        'md_epsilon': 1e-5,
        'kde_bw': 0.35
    }

    x_manifold_para['target_labels'] = None
    metrics_para[XManifoldEvaluation] = x_manifold_para

    qualified_metric_dict: dict = {
        CausalConstraintEvaluation: 'pdf_thr_0.9',
        GlobalImmutableEvaluation: 'global_immutable_0.005',
        GlobalMonotonicEvaluation: 'global_monotonic_0.005',
        # UserImmutableEvaluation: 'user_immutable_0.005',
        # UserCosineEvaluation: 'user_cos_0.5'
    }

    return metrics_para, qualified_metric_dict


