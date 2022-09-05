import math

import torch
from rllib.dataset.datatypes import Observation
from rllib.util.early_stopping import EarlyStopping
from rllib.util.training.utilities import get_target, get_model_validation_score, calibration_score, \
    sharpness
from tqdm import tqdm

from lib.model.bayesian_model import BayesianNNModel
from utils.logger import Logger


def _grad(model, prediction, target):
    y_batch = prediction
    y = target

    avg_log_likelihood = model.get_avg_log_likelihood(y_batch, y)
    prior_prob = model.get_prior_prob()
    num_train_samples = prediction[..., 0].numel()
    prior_pre_factor = 1 / math.sqrt(num_train_samples) if model.sqrt_mode else 1 / num_train_samples
    post_log_prob = avg_log_likelihood + prior_pre_factor * prior_prob

    score = torch.autograd.grad(post_log_prob.sum(), model.particles)[0]

    particles_copy = model.particles.detach().clone()
    k_xx = model.kernel(model.particles, particles_copy)  # (k, k)
    k_grad = torch.autograd.grad(k_xx.sum(), model.particles)[0]
    svgd_grads_stacked = k_xx @ score - k_grad / model.num_particles  # (k, n)

    return -svgd_grads_stacked, -post_log_prob.sum().detach()


def bayesian_model_grad(model, observation, dynamical_model=None):
    """Get bayesian model loss"""
    state, action = observation.state, observation.action

    normalized_state, normalized_action = model.normalize_state_and_action(
        state.reshape((-1, state.shape[-1])),
        action.reshape((-1, action.shape[-1]))
    )
    normalized_prediction = model.forward_nn(normalized_state, normalized_action)

    target = get_target(model, observation)
    normalized_target = model.normalize_target(target.reshape((-1, target.shape[-1])))

    return _grad(model, normalized_prediction, normalized_target)


def train_bayesian_nn_step(model, observation, optimizer, weight=1.0, dynamical_model=None):
    """Train a Neural Network Model."""
    optimizer.zero_grad()
    grad, loss = bayesian_model_grad(model, observation, dynamical_model=dynamical_model)
    model.particles.grad = grad
    optimizer.step()

    return loss


def _train_model_step(
    model, observation, optimizer, mask, logger, dynamical_model=None
):
    if not isinstance(observation, Observation):
        observation = Observation(**observation)
    observation.action = observation.action[..., : model.dim_action[0]]
    if isinstance(model, BayesianNNModel):
        loss = train_bayesian_nn_step(
            model, observation, optimizer, mask, dynamical_model=dynamical_model
        )
    else:
        raise TypeError("Functions for training Bayesian Neural Networks only.")
    logger.update(**{f"{model.model_kind[:3]}-loss": loss.item()})


def _validate_model_step(model, observation, logger, dynamical_model=None):
    if not isinstance(observation, Observation):
        observation = Observation(**observation)
    observation.action = observation.action[..., : model.dim_action[0]]
    _, mse, sharpness_, calibration_score_ = get_model_validation_score(
        model, observation, dynamical_model=dynamical_model
    )

    logger.update(
        **{
            f"{model.model_kind[:3]}-val-mse": mse,
            f"{model.model_kind[:3]}-sharp": sharpness_,
            f"{model.model_kind[:3]}-calib": calibration_score_,
        }
    )
    return mse


def train_model(
    model,
    train_set,
    optimizer,
    batch_size=100,
    num_epochs=None,
    max_iter=100,
    epsilon=0.1,
    min_iter=1,
    non_decrease_iter=float("inf"),
    logger=None,
    validation_set=None,
    dynamical_model=None,
):
    """Train a Predictive Model.

    Parameters
    ----------
    model: AbstractModel.
        Predictive model to optimize.
    train_set: ExperienceReplay.
        Dataset to train with.
    optimizer: Optimizer.
        Optimizer to call for the model.
    batch_size: int (default=1000).
        Batch size to iterate through.
    num_epochs: int, optional.
        Maximum number of epochs.
    max_iter: int (default = 100).
        Maximum number of iterations.
    min_iter: int (default=1).
        Minimum number of iterations before early stopping.
    epsilon: float.
        Early stopping parameter. If epoch loss is > (1 + epsilon) of minimum loss the
        optimization process stops.
    non_decrease_iter: int, optional.
        Early stopping parameter. If epoch loss does not decrease for consecutive
        non_decrease_iter, the optimization process stops.
    logger: Logger, optional.
        Progress logger.
    validation_set: ExperienceReplay, optional.
        Dataset to validate with.
    dynamical_model: AbstractModel, optional.
        Model to propagate predictions with.
    """
    if logger is None:
        logger = Logger(f"{model.name}_training")
    if validation_set is None:
        validation_set = train_set

    data_size = len(train_set) // batch_size

    if num_epochs is not None:
        max_iter = data_size * num_epochs
        min_iter = data_size * min_iter

    model.train()
    early_stopping = EarlyStopping(epsilon, non_decrease_iter=non_decrease_iter)

    for num_iter in tqdm(range(max_iter)):
        observation, idx, mask = train_set.sample_batch(batch_size)
        _train_model_step(
            model, observation, optimizer, mask, logger, dynamical_model=dynamical_model
        )

        observation, idx, mask = validation_set.sample_batch(batch_size)
        with torch.no_grad():
            mse = _validate_model_step(
                model, observation, logger, dynamical_model=dynamical_model
            )
        early_stopping.update(mse)

        if (
            (num_iter + 1) % data_size == 0
            and early_stopping.stop
            and num_iter > min_iter
        ):
            return


def calibrate_model(
    model,
    calibration_set,
    max_iter=100,
    epsilon=0.0001,
    temperature_range=(0.1, 100.0),
    logger=None,
):
    """Calibrate a model by scaling the temperature.

    First, find a suitable temperature by logarithmic search (increasing or decreasing).
    Then, find a reasonable temperature by binary search.
    """
    if logger is None:
        logger = Logger(f"{model.name}_calibration")

    observation = calibration_set.all_data
    observation.action = observation.action[..., : model.dim_action[0]]

    with torch.no_grad():
        initial_score = calibration_score(model, observation).item()
    initial_temperature = model.temperature

    # Increase temperature.
    model.temperature = initial_temperature.clone()
    score, temperature = initial_score, initial_temperature.clone()
    for _ in range(max_iter):
        if model.temperature > 2 * temperature_range[1]:
            break
        model.temperature *= 2
        with torch.no_grad():
            new_score = calibration_score(model, observation).item()
        if new_score > score:
            break
        score, temperature = new_score, model.temperature.clone()
    max_score, max_temperature = score, temperature

    # Decrease temperature.
    model.temperature = initial_temperature.clone()
    score, temperature = initial_score, initial_temperature.clone()
    for _ in range(max_iter):
        if model.temperature < temperature_range[0] / 2:
            break
        model.temperature /= 2
        with torch.no_grad():
            new_score = calibration_score(model, observation).item()
        if new_score > score:
            break
        score, temperature = new_score, model.temperature.clone()
    min_score, min_temperature = score, temperature

    if max_score < min_score:
        score, temperature = max_score, max_temperature
    else:
        score, temperature = min_score, min_temperature

    # Binary search:
    min_temperature, max_temperature = temperature / 2, 2 * temperature
    with torch.no_grad():
        model.temperature = max_temperature
        max_score = calibration_score(model, observation).item()
        model.temperature = min_temperature
        min_score = calibration_score(model, observation).item()

    if min_score > max_score:
        max_score, min_score = min_score, max_score
        max_temperature, min_temperature = min_temperature, max_temperature

    for _ in range(max_iter):
        if max_score - min_score < epsilon:
            break

        if score < max_score:
            max_score, max_temperature = score, temperature.clone()
        else:
            min_score, min_temperature = score, temperature.clone()

        if min_score > max_score:
            max_score, min_score = min_score, max_score
            max_temperature, min_temperature = min_temperature, max_temperature

        temperature = torch.exp(
            0.5 * (torch.log(min_temperature) + torch.log(max_temperature))
        )
        model.temperature = temperature.clone().clamp(*temperature_range)
        with torch.no_grad():
            score = calibration_score(model, observation).item()
    sharpness_ = sharpness(model, observation).item()

    logger.update(
        **{
            f"{model.model_kind[:3]}-temperature": model.temperature.item(),
            f"{model.model_kind[:3]}-post-sharp": sharpness_,
            f"{model.model_kind[:3]}-post-calib": score,
        }
    )


def evaluate_model(model, observation, logger=None, dynamical_model=None):
    """Train a Predictive Model.

    Parameters
    ----------
    model: AbstractModel.
        Predictive model to evaluate.
    observation: Observation.
        Observation to evaluate..
    logger: Logger, optional.
        Progress logger.
    dynamical_model: AbstractModel, optional.
        Model to propagate predictions with.
    """
    if logger is None:
        logger = Logger(f"{model.name}_evaluation")

    model.eval()
    with torch.no_grad():
        loss, mse, sharpness_, calibration_score_ = get_model_validation_score(
            model, observation, dynamical_model=dynamical_model
        )

        logger.update(
            **{
                f"{model.model_kind[:3]}-eval-loss": loss,
                f"{model.model_kind[:3]}-eval-mse": mse,
                f"{model.model_kind[:3]}-eval-sharp": sharpness_,
                f"{model.model_kind[:3]}-eval-calib": calibration_score_,
            }
        )
