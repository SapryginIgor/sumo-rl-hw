import torch
from torch import nn
from collections import deque
import numpy as np
from copy import copy

def create_network(input_dim, hidden_dims, output_dim):
    arr = []
    for hidden in hidden_dims:
        arr.append(nn.Linear(input_dim, hidden))
        arr.append(nn.ReLU())
        input_dim = hidden
    arr.append(nn.Linear(input_dim, output_dim))
    network = nn.Sequential(*arr)
    ##############################
    return network

def select_action_eps_greedy(Q, state, epsilon):
    """Выбирает действие epsilon-жадно."""
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32)
    Q_s = Q(state).detach().numpy()

    # action =
    ####### Здесь ваш код ########
    if torch.rand(1) < epsilon:
        action = np.random.randint(Q_s.shape[0])
    else:
        action = np.argmax(Q_s)
    ##############################

    action = int(action)
    return action

def to_tensor(x, dtype=np.float32):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=dtype)
    x = torch.from_numpy(x)
    return x

def compute_td_target(
        Q, rewards, next_states, terminateds, gamma=0.99, check_shapes=True,
):
    """ Считает TD-target."""

    # переводим входные данные в тензоры
    r = to_tensor(rewards)  # shape: [batch_size]
    s_next = to_tensor(next_states)  # shape: [batch_size, state_size]
    term = to_tensor(terminateds, bool)  # shape: [batch_size]

    # получаем Q[s_next, .] — значения полезности всех действий в следующем состоянии
    # Q_sn = ...,
    # а затем вычисляем V^*[s_next] — оптимальные значения полезности следующем состоянии
    # V_sn = ...
    ####### Здесь ваш код ########
    Q_sn = Q(s_next)

    V_sn = Q_sn.max(dim=-1)[0].detach()
    ##############################

    assert V_sn.dtype == torch.float32

    # вычисляем TD target
    # target = ...
    ####### Здесь ваш код ########
    target = r + gamma * (1 - term.to(torch.int32)) * V_sn
    ##############################

    if check_shapes:
        assert Q_sn.data.dim() == 2, \
            "убедитесь, что вы предсказали q-значения для всех действий в следующем состоянии"
        assert V_sn.data.dim() == 1, \
            "убедитесь, что вы вычислили V (s ') как максимум только по оси действий, а не по всем осям"
        assert target.data.dim() == 1, \
            "что-то не так с целевыми q-значениями, они должны быть вектором"

    return target
def compute_td_loss(
        Q, states, actions, td_target, regularizer=.1, out_non_reduced_losses=False
):
    """ Считает TD ошибку."""

    # переводим входные данные в тензоры
    s = to_tensor(states)  # shape: [batch_size, state_size]
    a = to_tensor(actions, int).long()  # shape: [batch_size]

    # получаем Q[s, a] для выбранных действий в текущих состояниях
    # (для каждого примера из батча)
    # Q_s_a = ...
    ####### Здесь ваш код ########
    Q_s_a = torch.gather(Q(s), -1, a.unsqueeze(-1)).squeeze(-1)
    ##############################

    # вычисляем TD error
    # td_error = ...
    ####### Здесь ваш код ########
    td_error = Q_s_a - td_target
    ##############################

    # MSE loss для минимизации
    td_losses = td_error ** 2
    loss = torch.mean(td_losses)
    # добавляем L1 регуляризацию на значения Q
    loss += regularizer * torch.abs(Q_s_a).mean()

    if out_non_reduced_losses:
        return loss, td_losses.detach()

    return loss


def linear(st, end, duration, t):
    """
    Линейная интерполяция значений в пределах диапазона [st, end],
    используя прогресс по времени t относительно всего отведенного
    времени duration.
    """

    if t >= duration:
        return end
    return st + (end - st) * (t / duration)


def sample_batch(replay_buffer, n_samples):
    # sample randomly `n_samples` samples from replay buffer
    # and split an array of samples into arrays:
    #    states, actions, rewards, next_states, terminateds
    # Use np.random.default_rng().choice for sampling
    ####### Здесь ваш код ########
    indices = np.random.default_rng().choice(np.arange(len(replay_buffer)), min(n_samples, len(replay_buffer)), replace=True)
    samples = [replay_buffer[i] for i in indices]
    states, actions, rewards, next_states, terminateds = zip(*samples)
    ##############################

    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(terminateds)

def symlog(x):
    """
    Compute symlog values for a vector `x`.
    It's an inverse operation for symexp.
    """
    return np.sign(x) * np.log(np.abs(x) + 1)

def softmax(xs, temp=1.):
    exp_xs = np.exp((xs - xs.max()) / temp)
    return exp_xs / exp_xs.sum()

def sample_prioritized_batch(replay_buffer, n_samples):
    # Sample randomly `n_samples` examples from replay buffer
    # weighting by priority (example's TD error) and split an array
    # of sample tuples into arrays:
    #    states, actions, rewards, next_states, terminateds
    # Also, keep samples' indices (into `indices`) to return them too!
    # Note that each sample in replay buffer is a tuple:
    #   (priority, state, action, reward, next_state, terminated)
    # Use
    ####### Здесь ваш код ########
    indices = np.random.default_rng().choice(np.arange(len(replay_buffer)), min(n_samples, len(replay_buffer)), replace=False, p=softmax(symlog(np.array([sample[0] for sample in replay_buffer]))))
    samples = [replay_buffer[i] for i in indices]
    loss, states, actions, rewards, next_states, terminateds = zip(*samples)
    ##############################

    batch = (
        # np.array(loss),
        np.array(states), np.array(actions), np.array(rewards),
        np.array(next_states), np.array(terminateds)
    )
    return batch, indices

def update_batch(replay_buffer, indices, batch, new_priority):
    """Updates batches with corresponding indices
    replacing their priority values."""
    states, actions, rewards, next_states, terminateds = batch

    for i in range(len(indices)):
        new_batch = (
            new_priority[i], states[i], actions[i], rewards[i],
            next_states[i], terminateds[i]
        )
        replay_buffer[indices[i]] = new_batch

def sort_replay_buffer(replay_buffer):
    """Sorts replay buffer to move samples with
    lesser priority to the beginning ==> they will be
    replaced with the new samples sooner."""
    new_rb = deque(maxlen=replay_buffer.maxlen)
    new_rb.extend(sorted(replay_buffer, key=lambda sample: sample[0]))
    return new_rb


def eval_dqn(env_copy, Q):
    """Оценка качества работы алгоритма на одном эпизоде"""
    env = env_copy
    s, _ = env.reset()
    done, ep_return = False, 0.

    while not done:
        # set epsilon = 0 to make an agent act greedy
        a = select_action_eps_greedy(Q, s, epsilon=0.)
        s_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        ep_return += r
        s = s_next

        if done:
            break

    return ep_return


def run_dqn_prioritized_rb(
        env,
        hidden_dims=(64, 64), lr=1e-3, gamma=0.99,
        eps_st=.1, eps_end=.01, eps_dur=.05, total_max_steps=30_000,
        train_schedule=1, replay_buffer_size=1_000_000, batch_size=32,
        # eval_schedule=1000, smooth_ret_window=5, success_ret=200.
):
    replay_buffer = deque(maxlen=replay_buffer_size)
    # eval_return_history = deque(maxlen=smooth_ret_window)
    # env_copy = copy(env)
    Q = create_network(
        input_dim=env.observation_space.shape[0], hidden_dims=hidden_dims,
        output_dim=env.action_space.n
    )
    opt = torch.optim.Adam(Q.parameters(), lr=lr)

    s, _ = env.reset()
    done = False

    for global_step in range(1, total_max_steps + 1):
        epsilon = linear(
            eps_st, eps_end, eps_dur * total_max_steps, global_step
        )
        a = select_action_eps_greedy(Q, s, epsilon=epsilon)
        s_next, r, terminated, truncated, _ = env.step(a)

        # Compute new sample loss (compute w/o gradients!)
        ####### Здесь ваш код ########
        with torch.no_grad():
          td_target = compute_td_target(Q, [r], [s_next], [terminated], gamma=gamma, check_shapes=False)
          loss = compute_td_loss(Q, [s], [a], td_target)
          loss = loss.detach().numpy()
        ##############################

        replay_buffer.append((loss, s, a, r, s_next, terminated))
        done = terminated or truncated

        if global_step % train_schedule == 0:
            train_batch, indices = sample_prioritized_batch(
                replay_buffer, batch_size
            )
            (
                states, actions, rewards,
                next_states, terminateds
            ) = train_batch

            opt.zero_grad()
            td_target = compute_td_target(Q, rewards, next_states, terminateds, gamma=gamma)
            loss, td_losses = compute_td_loss(Q, states, actions, td_target, out_non_reduced_losses=True)
            loss.backward()
            opt.step()

            update_batch(
                replay_buffer, indices, train_batch, td_losses.numpy()
            )

        # with much slower scheduler periodically re-sort replay buffer
        # such that it will overwrite the least important samples
        if global_step % (10 * train_schedule) == 0:
            replay_buffer = sort_replay_buffer(replay_buffer)

        # if global_step % eval_schedule == 0:
        #     eval_return = eval_dqn(env_copy, Q)
        #     eval_return_history.append(eval_return)
        #     avg_return = np.mean(eval_return_history)
        #     print(f'{global_step=} | {avg_return=:.3f} | {epsilon=:.3f}')
        #     # if avg_return >= success_ret:
        #     #     print('Решено!')
        #     #     break

        s = s_next
        if done:
            s, _ = env.reset()
            done = False