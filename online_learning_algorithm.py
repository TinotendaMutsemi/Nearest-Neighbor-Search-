
import numpy as np

def online_learning_algorithm(Hn_t_plus_1, xt, bt, Hn_t, Sn_t_minus_1, St_minus_1, rule_g):
    """
    Implements the Online-Learning Algorithm (OLA).

    Args:
    Hn_t_plus_1: Updated agent-controls.
    xt: Current feature realization.
    bt: Current portfolio controls.
    Hn_t: Current agent-controls.
    Sn_t_minus_1: Past agent-wealth.
    St_minus_1: Past portfolio wealth.
    rule_g: Function for updating agent mixture.

    Returns:
    bt_plus_1: Updated portfolio controls.
    Sn_t: Updated agent-wealth.
    St: Updated portfolio wealth.
    qn_t_plus_1: Updated agent mixture.
    """

    # Initialize variables
    St = St_minus_1
    Sn_t = Sn_t_minus_1
    qn_t = np.ones(len(Hn_t)) / len(Hn_t)  # Initial agent mixture (uniform distribution)

    # Update portfolio wealth
    St = St_minus_1 * (bt.dot(xt - 1) + 1)

    # Update agent wealth
    Sn_t = Sn_t_minus_1 * (Hn_t.dot(xt - 1) + 1)

    # Update agent mixture
    qn_t_plus_1 = rule_g(qn_t, Sn_t)

    # Re-normalize agent mixtures
    if np.sum(qn_t_plus_1) != 1:
        qn_t_plus_1 = qn_t_plus_1 / np.sum(qn_t_plus_1)
    elif np.sum(np.abs(qn_t_plus_1)) != 1:
        qn_t_plus_1 = qn_t_plus_1 / np.sum(np.abs(qn_t_plus_1))

    # Update portfolio controls
    bt_plus_1 = np.sum(qn_t_plus_1[:, np.newaxis] * Hn_t_plus_1, axis=0)

    # Leverage corrections
    leverage = np.sum(np.abs(bt_plus_1))
    if leverage != 1:
        bt_plus_1 = bt_plus_1 / leverage
        qn_t_plus_1 = qn_t_plus_1 / leverage

    return bt_plus_1, Sn_t, St, qn_t_plus_1
