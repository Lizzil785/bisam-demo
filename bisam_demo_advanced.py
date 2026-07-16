import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import copy
import json
import math

COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
    '#1abc9c', '#e67e22', '#95a5a6', '#e91e63', '#00bcd4',
    '#ff5722', '#607d8b', '#795548', '#cddc39', '#ff9800'
]
DEFAULT_LR = {'SGD': 0.05, 'Adam': 0.01, 'AdamW': 0.01}
EPS = 1e-12
AUTO_PLAY_BASE_INTERVAL = 0.15
MAX_PREVIEW_HISTORY = 5

TEST_FUNCTIONS = {
    "二次凸函数": {
        "fn": lambda x, y: x**2 + y**2,
        "x_range": (-3, 3), "y_range": (-3, 3),
        "optima": [(0, 0)],
        "desc": "简单的凸函数，全局最小值在原点 (0, 0)，损失值为 0。优化器应能稳定、快速地收敛到原点。",
        "condition_number": "1（各向同性，最简单）"
    },
    "Rosenbrock": {
        "fn": lambda x, y: (1 - x)**2 + 100*(y - x**2)**2,
        "x_range": (-2, 3), "y_range": (-1, 4),
        "optima": [(1, 1)],
        "desc": "经典的香蕉形非凸函数，全局最小值在 (1, 1)，损失值为 0。其狭窄弯曲的谷底对优化器是重大考验，容易在谷壁间震荡。",
        "condition_number": "极高（≈100+），窄谷地形"
    },
    "Beale": {
        "fn": lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2,
        "x_range": (-4.5, 4.5), "y_range": (-4.5, 4.5),
        "optima": [(3, 0.5)],
        "desc": "多峰复杂函数，全局最小值在 (3, 0.5)，损失值为 0。存在多个局部极小和鞍点，对优化器的全局探索能力要求较高。",
        "condition_number": "高（高阶非线性，远离原点时梯度剧烈）"
    },
    "Himmelblau": {
        "fn": lambda x, y: (x**2 + y - 11)**2 + (x + y**2 - 7)**2,
        "x_range": (-5, 5), "y_range": (-5, 5),
        "optima": [(3, 2), (-2.8051, 3.1313), (-3.7793, -3.2832), (3.5844, -1.8481)],
        "desc": "具有 4 个完全相同的全局最小值（损失值均为 0），优化器可能收敛到其中任意一个，取决于初始位置。",
        "condition_number": "中等（多模态，各极小点附近条件数不同）"
    },
    "Ackley": {
        "fn": lambda x, y: -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))) + np.e + 20,
        "x_range": (-5, 5), "y_range": (-5, 5),
        "optima": [(0, 0)],
        "desc": "多峰波纹函数，全局最小值在 (0, 0)，损失值为 0。外围近似球形，内部有大量周期性局部极小，考验优化器的探索-利用平衡。",
        "condition_number": "复杂（周期性多峰，中心附近梯度平坦）"
    },
    "噪声二次函数": {
        "fn": lambda x, y: x**2 + y**2 + 0.3 * np.sin(5*x) * np.cos(5*y),
        "x_range": (-3, 3), "y_range": (-3, 3),
        "optima": [(0, 0)],
        "desc": "在二次函数基础上叠加高频振荡噪声，全局最小值仍在原点附近。考验优化器在噪声干扰下的鲁棒收敛能力。",
        "condition_number": "中等（基础条件数1 + 高频扰动）"
    },
    "鞍点函数": {
        "fn": lambda x, y: x**2 - y**2 + 0.5 * np.sin(3*x) + 0.5 * np.cos(3*y),
        "x_range": (-3, 3), "y_range": (-3, 3),
        "optima": [],
        "desc": "中心区域存在鞍点结构（x 方向凸、y 方向凹），叠加周期项产生多个局部极小。考验优化器逃离鞍点的能力。",
        "condition_number": "不定（Hessian 有正有负特征值）"
    },
}

TORCH_FUNCS = {
    "Ackley": lambda x, y: -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x**2 + y**2))) - torch.exp(0.5 * (torch.cos(2*math.pi*x) + torch.cos(2*math.pi*y))) + math.e + 20,
    "噪声二次函数": lambda x, y: x**2 + y**2 + 0.3 * torch.sin(5*x) * torch.cos(5*y),
    "鞍点函数": lambda x, y: x**2 - y**2 + 0.5 * torch.sin(3*x) + 0.5 * torch.cos(3*y),
}

GRID_RESOLUTION_HIGH = 200
GRID_RESOLUTION_LOW = 80

FUNC_CONVERGENCE_TIPS = {
    "二次凸函数": {
        "easy": "✅ 二次凸函数是最简单的优化场景，任何优化器都应能快速收敛。",
        "diverge": "❌ 即使在凸函数上发散，说明学习率过大，建议降至 0.01 以下。"
    },
    "Rosenbrock": {
        "easy": "✅ 在 Rosenbrock 上表现良好，说明优化器具备处理窄谷地形的能力。",
        "diverge": "❌ Rosenbrock 的窄谷容易导致梯度爆炸，建议减小学习率或使用自适应优化器。"
    },
    "Beale": {
        "easy": "✅ 在 Beale 函数上成功收敛，说明优化器具有较强的非线性优化能力。",
        "diverge": "❌ Beale 函数的高阶项容易引发数值不稳定，建议减小学习率并缩小 ρ。"
    },
    "Himmelblau": {
        "easy": "✅ 在 Himmelblau 上找到某个最小值，优化器多模态搜索能力正常。",
        "diverge": "❌ Himmelblau 的梯度变化剧烈，建议降低学习率或改用 Adam/AdamW。"
    },
    "Ackley": {
        "easy": "✅ 在 Ackley 上成功收敛到全局最优，说明优化器的探索-利用平衡良好。",
        "diverge": "❌ Ackley 的周期性波纹容易困住优化器，建议增大 ρ 或使用自适应模式。"
    },
    "噪声二次函数": {
        "easy": "✅ 在噪声干扰下仍能收敛，优化器鲁棒性良好。",
        "diverge": "❌ 噪声放大了梯度波动，建议减小学习率并适当增大 ρ 以平滑噪声。"
    },
    "鞍点函数": {
        "easy": "✅ 成功逃离鞍点区域，优化器具备较好的鞍点检测与逃逸能力。",
        "diverge": "❌ 鞍点附近梯度方向混乱，建议使用自适应模式让 BiSAM 自动调整扰动方向。"
    }
}

PRESET_CONFIGS = {
    "二次凸函数": {'base_opt': 'SGD', 'lr': 0.05, 'rho': 0.3, 'adaptive': True, 'update_freq': 5, 'loss_type': 'tanh', 'alpha': 0.1, 'mu': 1.0, 'init_x': 2.5, 'init_y': 2.5},
    "Rosenbrock": {'base_opt': 'Adam', 'lr': 0.01, 'rho': 0.5, 'adaptive': True, 'update_freq': 3, 'loss_type': 'tanh', 'alpha': 0.1, 'mu': 1.0, 'init_x': -1.0, 'init_y': 1.0},
    "Beale": {'base_opt': 'AdamW', 'lr': 0.005, 'rho': 0.3, 'adaptive': True, 'update_freq': 5, 'loss_type': 'log', 'alpha': 0.1, 'mu': 1.0, 'init_x': 1.0, 'init_y': 1.0},
    "Himmelblau": {'base_opt': 'Adam', 'lr': 0.01, 'rho': 0.5, 'adaptive': True, 'update_freq': 5, 'loss_type': 'tanh', 'alpha': 0.1, 'mu': 1.0, 'init_x': 0.0, 'init_y': 0.0},
    "Ackley": {'base_opt': 'AdamW', 'lr': 0.01, 'rho': 0.8, 'adaptive': True, 'update_freq': 3, 'loss_type': 'tanh', 'alpha': 0.05, 'mu': 1.0, 'init_x': 3.0, 'init_y': 3.0},
    "噪声二次函数": {'base_opt': 'SGD', 'lr': 0.03, 'rho': 0.5, 'adaptive': True, 'update_freq': 5, 'loss_type': 'tanh', 'alpha': 0.1, 'mu': 1.0, 'init_x': 2.0, 'init_y': 2.0},
    "鞍点函数": {'base_opt': 'Adam', 'lr': 0.01, 'rho': 0.6, 'adaptive': True, 'update_freq': 3, 'loss_type': 'log', 'alpha': 0.1, 'mu': 1.0, 'init_x': 0.1, 'init_y': 0.1},
}


# ==================== BiSAM 优化器 ====================
class BiSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.5, adaptive=True,
                 update_freq=5, alpha=0.1, mu=1.0, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, update_freq=update_freq,
                        alpha=alpha, mu=mu, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self._first_step_called = False
        self._step_count = 0

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        if self._first_step_called:
            raise RuntimeError("first_step already called")
        grad_norm_adaptive_sq = 0.0
        grad_norm_plain_sq = 0.0
        for group in self.param_groups:
            adaptive = group['adaptive']
            for p in group['params']:
                if p.grad is None:
                    continue
                if adaptive:
                    g = p.grad * p.abs()
                    grad_norm_adaptive_sq += g.norm(p=2).item() ** 2
                else:
                    grad_norm_plain_sq += p.grad.norm(p=2).item() ** 2
        grad_norm_adaptive = grad_norm_adaptive_sq ** 0.5
        grad_norm_plain = grad_norm_plain_sq ** 0.5
        step_count = self._step_count

        for group in self.param_groups:
            rho = group['rho']
            adaptive = group['adaptive']
            update_freq = group['update_freq']
            compute_perturb = (step_count % update_freq == 0)
            group_grad_norm = grad_norm_adaptive if adaptive else grad_norm_plain
            base_scale = rho / (group_grad_norm + EPS)
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['old_p'] = p.data.clone()
                if compute_perturb or 'd' not in state:
                    perturb_dir = p.grad.clone()
                    if adaptive:
                        p_norm = p.norm(p=2)
                        scale = base_scale * p_norm if p_norm > EPS else base_scale
                    else:
                        scale = base_scale
                    dir_norm = perturb_dir.norm(p=2)
                    d = perturb_dir / dir_norm if dir_norm > EPS else torch.zeros_like(perturb_dir)
                    state['d'] = d
                else:
                    d = state['d']
                    if adaptive:
                        p_norm = p.norm(p=2)
                        scale = base_scale * p_norm if p_norm > EPS else base_scale
                    else:
                        scale = base_scale
                state['perturb_size'] = scale * d.norm(p=2)
                p.add_(d, alpha=scale)
        self._first_step_called = True
        self._step_count = step_count + 1
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        if not self._first_step_called:
            raise RuntimeError("first_step must be called first")
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state.get(p)
                if state and 'old_p' in state:
                    p.data = state['old_p']
        self.base_optimizer.step()
        self._first_step_called = False
        if zero_grad:
            self.zero_grad()

    def state_dict(self):
        base_state = self.base_optimizer.state_dict()
        bi_sam_state = super().state_dict()
        return {'base_optimizer': base_state, 'bi_sam_state': bi_sam_state,
                'first_step_called': self._first_step_called,
                'step_count': self._step_count}

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        super().load_state_dict(state_dict['bi_sam_state'])
        self._first_step_called = state_dict['first_step_called']
        self._step_count = state_dict.get('step_count', 0)
        self.param_groups = self.base_optimizer.param_groups


# ==================== 下界损失 ====================
def bisam_loss_tanh(pred, target, alpha=0.1):
    return 1.0 - torch.tanh(alpha * (pred - target))

def bisam_loss_log(pred, target, mu=1.0):
    return 1.0 - F.softplus((pred - target) - mu)


# ==================== 学习率/ρ 调度 ====================
def get_schedule_factor(scheduler_type, step, total_steps):
    if total_steps <= 1:
        return 1.0
    progress = step / (total_steps - 1)
    if scheduler_type == "none":
        return 1.0
    elif scheduler_type == "cosine":
        return 0.5 * (1 + math.cos(math.pi * progress))
    elif scheduler_type == "linear":
        return max(1.0 - progress, 0.01)
    elif scheduler_type == "warmup_cosine":
        warmup_ratio = 0.1
        if progress < warmup_ratio:
            return progress / warmup_ratio
        else:
            t = (progress - warmup_ratio) / (1 - warmup_ratio)
            return 0.5 * (1 + math.cos(math.pi * t))
    return 1.0


# ==================== 测试函数与网格缓存 ====================
@st.cache_resource(show_spinner=False)
def get_test_grid(name, resolution):
    info = TEST_FUNCTIONS[name]
    x_range = np.linspace(*info["x_range"], resolution)
    y_range = np.linspace(*info["y_range"], resolution)
    X, Y = np.meshgrid(x_range, y_range)
    Z = info["fn"](X, Y)
    return info["fn"], X, Y, x_range, y_range, Z


def compute_perturb_size(optimizer):
    sizes = [
        optimizer.state[p]['perturb_size'].item()
        for group in optimizer.param_groups
        for p in group['params']
        if p in optimizer.state and 'perturb_size' in optimizer.state[p]
    ]
    return np.mean(sizes) if sizes else 0.0


# ==================== 单次运行 ====================
# ... existing code ...
def run_optimization(config, steps=50):
    resolution = GRID_RESOLUTION_HIGH
    f_np, X, Y, x_range, y_range, Z = get_test_grid(config['func'], resolution)
    f_torch = TORCH_FUNCS.get(config['func'], f_np)

    if not np.isfinite(config['init_x']) or not np.isfinite(config['init_y']):
        raise ValueError(f"初始值无效: ({config['init_x']}, {config['init_y']})，请输入有限数值")

    param = torch.tensor([config['init_x'], config['init_y']], dtype=torch.float32, requires_grad=True)
    base_cls = {'SGD': SGD, 'Adam': Adam, 'AdamW': AdamW}[config['base_opt']]
    optimizer = BiSAM([param], base_cls, lr=config['lr'], rho=config['rho'],
                      adaptive=config['adaptive'], update_freq=config['update_freq'],
                      alpha=config['alpha'], mu=config['mu'])

    pos_arr = np.zeros((steps + 1, 2), dtype=np.float64)
    loss_arr = np.zeros(steps + 1, dtype=np.float64)
    pos_arr[0] = [param[0].item(), param[1].item()]
    loss_arr[0] = float(f_np(param[0].item(), param[1].item()))
    grad_norm_list = []
    perturb_list = []
    target_zero = torch.tensor(0.0)

    base_lr = config['lr']
    base_rho = config['rho']
    lr_scheduler = config.get('lr_scheduler', 'none')
    rho_scheduler = config.get('rho_scheduler', 'none')
    grad_clip = config.get('grad_clip', 0.0)

    for i in range(steps):
        lr_factor = get_schedule_factor(lr_scheduler, i, steps)
        rho_factor = get_schedule_factor(rho_scheduler, i, steps)
        for pg in optimizer.base_optimizer.param_groups:
            pg['lr'] = base_lr * lr_factor
        for pg in optimizer.param_groups:
            pg['rho'] = base_rho * rho_factor

        optimizer.zero_grad()
        loss_val = f_torch(param[0], param[1])
        if config['loss_type'] == 'tanh':
            inner_loss = bisam_loss_tanh(loss_val, target_zero, config['alpha'])
        else:
            inner_loss = bisam_loss_log(loss_val, target_zero, config['mu'])
        inner_loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([param], grad_clip)

        if not torch.isfinite(param.grad).all():
            raise ValueError(f"第 {i+1} 步梯度出现 NaN/Inf，请尝试减小学习率或调整初始位置")

        grad_norm_list.append(param.grad.norm().item())
        optimizer.first_step()
        perturb_list.append(compute_perturb_size(optimizer))

        optimizer.zero_grad()
        loss_val_perturbed = f_torch(param[0], param[1])
        loss_val_perturbed.backward()
        optimizer.second_step()

        loss_after = f_torch(param[0], param[1])

        pos_arr[i + 1] = [param[0].item(), param[1].item()]
        loss_arr[i + 1] = float(loss_after.item())

        if not np.isfinite(loss_arr[i + 1]):
            raise ValueError(f"第 {i+1} 步损失值发散 (={loss_arr[i+1]:.2e})，请尝试减小学习率或调整初始位置")

    return {
        'pos': pos_arr,
        'loss': loss_arr,
        'grad_norm': np.array(grad_norm_list),
        'perturb': np.array(perturb_list),
        'config': config
    }


def _get_grid_for_result(res, resolution=GRID_RESOLUTION_HIGH):
    _, X, Y, x_range, y_range, Z = get_test_grid(res['config']['func'], resolution)
    return X, Y, x_range, y_range, Z


# ==================== 分析函数 ====================
def _nearest_optima(pos, func_name):
    optima = TEST_FUNCTIONS[func_name].get('optima', [])
    if not optima:
        return None, None
    dists = [np.sqrt((pos[0] - ox)**2 + (pos[1] - oy)**2) for ox, oy in optima]
    best_idx = int(np.argmin(dists))
    return optima[best_idx], dists[best_idx]


def _convergence_phase(loss_arr):
    n = len(loss_arr)
    if n < 4:
        return "数据不足"
    q1 = n // 4
    q2 = n // 2
    q3 = 3 * n // 4
    avg_early = np.mean(loss_arr[:q1])
    avg_mid = np.mean(loss_arr[q1:q2])
    avg_late = np.mean(loss_arr[q3:])
    if avg_early > avg_mid > avg_late and (avg_early - avg_late) > 1e-6:
        total_drop = loss_arr[0] - loss_arr[-1]
        first_half_drop = loss_arr[0] - loss_arr[q2]
        ratio = first_half_drop / total_drop if abs(total_drop) > 1e-10 else 0.5
        if ratio > 0.7:
            return "前期快速收敛"
        elif ratio > 0.4:
            return "均匀收敛"
        else:
            return "后期精调收敛"
    elif abs(avg_early - avg_late) < 1e-6:
        return "基本未收敛（损失平稳）"
    else:
        return "震荡/不稳定"


def _distance_traveled(pos_arr):
    diffs = np.diff(pos_arr, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))


def _ema_smooth(data, alpha=0.15):
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def _score_run(result):
    loss = result['loss']
    pos = result['pos']
    grad = result['grad_norm']
    perturb = result['perturb']
    func_name = result['config']['func']

    loss_drop_pct = ((loss[0] - loss[-1]) / max(abs(loss[0]), 1e-10)) * 100
    speed_score = min(loss_drop_pct / 100, 1.0)

    final_loss = loss[-1]
    precision_score = max(1.0 - final_loss / max(abs(loss[0]), 1.0), 0.0)

    total_dist = _distance_traveled(pos)
    straight = np.sqrt((pos[-1, 0] - pos[0, 0])**2 + (pos[-1, 1] - pos[0, 1])**2)
    efficiency = straight / total_dist if total_dist > 1e-10 else 0.0

    late_std = np.std(loss[-max(len(loss)//5, 3):])
    stability_score = max(1.0 - late_std / max(abs(loss[0]), 1e-10) * 100, 0.0)
    stability_score = min(stability_score, 1.0)

    perturb_mean = np.mean(perturb)
    if perturb_mean > 0.01:
        perturb_util_score = min(perturb_mean / 1.0, 1.0)
    else:
        perturb_util_score = perturb_mean / 0.01

    optima = TEST_FUNCTIONS[func_name].get('optima', [])
    if optima:
        _, dist_opt = _nearest_optima(pos[-1], func_name)
        if dist_opt is not None:
            proximity_score = max(1.0 - dist_opt / 5.0, 0.0)
        else:
            proximity_score = 0.5
    else:
        proximity_score = precision_score

    return {
        '收敛速度': speed_score,
        '最终精度': precision_score,
        '路径效率': efficiency,
        '稳定性': stability_score,
        '扰动利用': perturb_util_score,
        '最优接近': proximity_score,
    }


def generate_single_analysis(result):
    cfg = result['config']
    loss = result['loss']
    pos = result['pos']
    grad = result['grad_norm']
    perturb = result['perturb']
    func_name = cfg['func']

    loss_drop = loss[0] - loss[-1]
    loss_drop_pct = (loss_drop / loss[0] * 100) if abs(loss[0]) > 1e-10 else 0.0
    nearest_opt, dist_opt = _nearest_optima(pos[-1], func_name)
    phase = _convergence_phase(loss)
    total_dist = _distance_traveled(pos)
    grad_mean = np.mean(grad)
    grad_final = grad[-1] if len(grad) > 0 else 0.0
    perturb_mean = np.mean(perturb)
    perturb_max = np.max(perturb) if len(perturb) > 0 else 0.0
    loss_std_late = np.std(loss[-max(len(loss)//5, 3):])

    lines = []
    lines.append(f"#### 📝 优化过程分析")

    lines.append(f"**测试函数：** {func_name} — {TEST_FUNCTIONS[func_name]['desc']}")
    lines.append(f"**Hessian 条件数：** {TEST_FUNCTIONS[func_name].get('condition_number', '未知')}")

    lines.append(f"**收敛阶段判断：** {phase}")
    if "快速" in phase:
        lines.append("> 大部分损失下降发生在优化前半段，说明当前参数配置下 BiSAM 在初始阶段就能捕获大量梯度信息并高效利用。")
    elif "均匀" in phase:
        lines.append("> 损失在整个优化过程中均匀下降，优化器在各阶段都保持了稳定的更新力度。")
    elif "后期" in phase:
        lines.append("> 优化器在前半段进展较慢，后半段加速收敛。这可能与 BiSAM 的扰动探索机制有关——前期探索、后期集中利用。")
    elif "震荡" in phase:
        lines.append("> 损失曲线出现明显震荡，可能是学习率偏大或 ρ 设置过高导致扰动过激。建议适当减小学习率或 ρ。")
    elif "未收敛" in phase:
        lines.append("> 损失几乎未变化，优化器可能陷入了平台区或学习率过小。建议增大学习率或更换初始位置。")

    lines.append(f"**损失变化：** 初始 {loss[0]:.4f} → 最终 {loss[-1]:.6f}（下降 {loss_drop:.4f}，降幅 {loss_drop_pct:.1f}%）")
    if loss_drop_pct > 90:
        lines.append("> 🎯 降幅超过 90%，优化效果优秀。")
    elif loss_drop_pct > 50:
        lines.append("> 👍 降幅在 50%-90% 之间，优化效果良好。")
    elif loss_drop_pct > 10:
        lines.append("> ⚠️ 降幅在 10%-50% 之间，仍有较大优化空间，可尝试更多步数或调整参数。")
    else:
        lines.append("> ❗ 降幅不足 10%，优化效果不佳，建议检查学习率、ρ 或初始位置。")

    if nearest_opt is not None:
        lines.append(f"**与最近最优解的距离：** 终点 {pos[-1][0]:.4f}, {pos[-1][1]:.4f}，距最近最优点 ({nearest_opt[0]}, {nearest_opt[1]}) 距离 = {dist_opt:.4f}")
        if dist_opt < 0.1:
            lines.append("> ✅ 已非常接近理论最优解，优化器成功找到了全局最小值区域。")
        elif dist_opt < 1.0:
            lines.append("> 🔶 接近最优解但仍有差距，可继续增加优化步数或微调参数以进一步逼近。")
        elif dist_opt < 3.0:
            lines.append("> ⚠️ 距离最优解较远，可能陷入了局部极小或鞍点附近。")
        else:
            lines.append("> ❌ 远离任何已知最优点，优化基本失败。建议大幅调整参数。")

    lines.append(f"**路径分析：** 总移动距离 = {total_dist:.2f}，平均梯度范数 = {grad_mean:.4f}，最终梯度范数 = {grad_final:.6f}")
    straight_dist = np.sqrt((pos[-1, 0] - pos[0, 0])**2 + (pos[-1, 1] - pos[0, 1])**2)
    efficiency = straight_dist / total_dist if total_dist > 1e-10 else 0.0
    lines.append(f"**路径效率：** 直线距离 = {straight_dist:.2f}，效率 = {efficiency:.1%}")
    if efficiency > 0.7:
        lines.append("> 📈 路径效率很高（>70%），优化器基本沿直线向最优解前进，方向判断准确。")
    elif efficiency > 0.4:
        lines.append("> 📊 路径效率中等（40%-70%），存在一定程度的绕行，这在复杂地形中是正常的。")
    else:
        lines.append("> 🔄 路径效率较低（<40%），优化器走了大量弯路。可能是 BiSAM 的扰动探索导致的震荡，建议减小 ρ。")

    lines.append(f"**扰动分析：** 平均扰动幅度 = {perturb_mean:.6f}，最大扰动幅度 = {perturb_max:.6f}")
    if perturb_mean > 1.0:
        lines.append("> ⚡ 扰动幅度较大，BiSAM 在每步进行了大幅度的邻域探索。这有助于跳出局部极小，但也可能导致不稳定。")
    elif perturb_mean > 0.1:
        lines.append("> 🔧 扰动幅度适中，在探索和利用之间取得了较好的平衡。")
    else:
        lines.append("> 🔬 扰动幅度很小，BiSAM 的邻域搜索效果有限，接近普通优化器行为。可适当增大 ρ。")

    if loss_std_late < 1e-4:
        lines.append("**收敛稳定性：** ✅ 后期损失波动极小（σ < 1e-4），优化器已稳定收敛。")
    elif loss_std_late < 1e-2:
        lines.append("**收敛稳定性：** 🔶 后期存在轻微波动，但整体趋势稳定。")
    else:
        lines.append("**收敛稳定性：** ❗ 后期损失波动较大，优化尚未完全稳定，建议增加步数或减小学习率。")

    sched_info = []
    lr_sched = cfg.get('lr_scheduler', 'none')
    rho_sched = cfg.get('rho_scheduler', 'none')
    if lr_sched != 'none':
        sched_info.append(f"学习率调度={lr_sched}")
    if rho_sched != 'none':
        sched_info.append(f"ρ调度={rho_sched}")
    grad_clip = cfg.get('grad_clip', 0.0)
    if grad_clip > 0:
        sched_info.append(f"梯度裁剪={grad_clip}")
    if sched_info:
        lines.append(f"**调度与裁剪：** {' | '.join(sched_info)}")

    tips = FUNC_CONVERGENCE_TIPS.get(func_name, {})
    if loss[-1] < 1.0:
        lines.append(f"\n{tips.get('easy', '')}")
    else:
        lines.append(f"\n{tips.get('diverge', '')}")

    return "\n\n".join(lines)


def generate_diff_analysis(prev_result, prev_config, curr_result, curr_config):
    lines = []
    lines.append("#### 🔍 参数变化影响分析")

    changes = []
    if prev_config['base_opt'] != curr_config['base_opt']:
        changes.append(f"基础优化器从 **{prev_config['base_opt']}** 切换为 **{curr_config['base_opt']}**")
    if prev_config['lr'] != curr_config['lr']:
        direction = "增大" if curr_config['lr'] > prev_config['lr'] else "减小"
        changes.append(f"学习率{direction}：{prev_config['lr']:.4f} → {curr_config['lr']:.4f}")
    if prev_config['rho'] != curr_config['rho']:
        direction = "增大" if curr_config['rho'] > prev_config['rho'] else "减小"
        changes.append(f"ρ{direction}：{prev_config['rho']:.2f} → {curr_config['rho']:.2f}")
    if prev_config['adaptive'] != curr_config['adaptive']:
        changes.append(f"自适应模式：{'开启' if curr_config['adaptive'] else '关闭'}（之前为{'开启' if prev_config['adaptive'] else '关闭'}）")
    if prev_config['update_freq'] != curr_config['update_freq']:
        changes.append(f"更新频率 K：{prev_config['update_freq']} → {curr_config['update_freq']}")
    if prev_config['loss_type'] != curr_config['loss_type']:
        changes.append(f"下界损失类型从 **{prev_config['loss_type']}** 改为 **{curr_config['loss_type']}**")
    if prev_config['init_x'] != curr_config['init_x'] or prev_config['init_y'] != curr_config['init_y']:
        changes.append(f"起始位置从 ({prev_config['init_x']}, {prev_config['init_y']}) 移至 ({curr_config['init_x']}, {curr_config['init_y']})")
    if prev_config.get('lr_scheduler', 'none') != curr_config.get('lr_scheduler', 'none'):
        changes.append(f"学习率调度：{prev_config.get('lr_scheduler', 'none')} → {curr_config.get('lr_scheduler', 'none')}")
    if prev_config.get('rho_scheduler', 'none') != curr_config.get('rho_scheduler', 'none'):
        changes.append(f"ρ调度：{prev_config.get('rho_scheduler', 'none')} → {curr_config.get('rho_scheduler', 'none')}")

    if not changes:
        lines.append("参数未发生变化，两次运行结果相同。")
        return "\n\n".join(lines)

    lines.append("**变更内容：**")
    for c in changes:
        lines.append(f"- {c}")

    prev_final = prev_result['loss'][-1]
    curr_final = curr_result['loss'][-1]
    delta = curr_final - prev_final
    lines.append(f"\n**效果对比：** 最终损失从 {prev_final:.6f} 变为 {curr_final:.6f}（{'改善' if delta < 0 else '恶化'} {abs(delta):.6f}）")

    if delta < -0.01:
        lines.append("> 📈 本次参数调整带来了**显著改善**。")
    elif delta < -1e-4:
        lines.append("> 📈 本次参数调整带来了**轻微改善**。")
    elif delta < 1e-4:
        lines.append("> ➡️ 两次结果基本持平，参数变化影响不大。")
    elif delta < 0.01:
        lines.append("> 📉 本次参数调整导致了**轻微恶化**。")
    else:
        lines.append("> 📉 本次参数调整导致了**显著恶化**，建议回退到之前的配置。")

    if any("学习率" in c for c in changes):
        if curr_config['lr'] > prev_config['lr'] and delta > 0:
            lines.append("> 💡 学习率增大导致性能下降，建议回退到较小的学习率。")
        elif curr_config['lr'] < prev_config['lr'] and delta > 0:
            lines.append("> 💡 学习率过小导致收敛不充分，可适当增大学习率。")
        elif curr_config['lr'] > prev_config['lr'] and delta < 0:
            lines.append("> 💡 增大学习率加速了收敛，当前学习率更合适。")
        elif curr_config['lr'] < prev_config['lr'] and delta < 0:
            lines.append("> 💡 减小学习率提升了稳定性，当前学习率更合适。")

    if any("ρ" in c and "调度" not in c for c in changes):
        if curr_config['rho'] > prev_config['rho'] and delta < 0:
            lines.append("> 💡 增大 ρ 增强了 BiSAM 的邻域探索能力，有助于找到更优解。")
        elif curr_config['rho'] > prev_config['rho'] and delta > 0:
            lines.append("> 💡 增大 ρ 反而导致性能下降，扰动过大可能破坏了优化稳定性。")
        elif curr_config['rho'] < prev_config['rho'] and delta < 0:
            lines.append("> 💡 减小 ρ 提升了稳定性，当前配置在利用方面更出色。")

    if any("自适应" in c for c in changes):
        if curr_config['adaptive'] and delta < 0:
            lines.append("> 💡 开启自适应模式改善了优化效果，说明参数相关的梯度缩放是有益的。")
        elif not curr_config['adaptive'] and delta < 0:
            lines.append("> 💡 关闭自适应模式反而更好，可能当前函数的梯度分布不需要参数相关的缩放。")

    return "\n\n".join(lines)


def generate_compare_analysis(results):
    lines = []
    lines.append("#### 🏆 综合分析与排名")

    n = len(results)
    if n < 2:
        lines.append("至少需要 2 个成功运行的配置才能进行对比分析。")
        return "\n\n".join(lines)

    final_losses = [r['loss'][-1] for r in results]
    min_losses = [r['loss'].min() for r in results]
    total_dists = [_distance_traveled(r['pos']) for r in results]
    phases = [_convergence_phase(r['loss']) for r in results]

    rank_final = np.argsort(final_losses)
    rank_min = np.argsort(min_losses)

    lines.append("**🥇 最终损失排名（越低越好）：**")
    for rank, idx in enumerate(rank_final):
        medal = ["🥇", "🥈", "🥉"] + ["  "] * (n - 3)
        cfg = results[idx]['config']
        lines.append(f"{medal[rank]} **Run {idx+1}** — 最终损失 = {final_losses[idx]:.6f} | "
                      f"{cfg['base_opt']}, lr={cfg['lr']}, ρ={cfg['rho']}, {cfg['func']}")

    best_idx = rank_final[0]
    worst_idx = rank_final[-1]
    best_cfg = results[best_idx]['config']
    worst_cfg = results[worst_idx]['config']
    gap = final_losses[worst_idx] - final_losses[best_idx]

    lines.append(f"\n**关键发现：**")
    lines.append(f"- 最佳配置为 **Run {best_idx+1}**（{best_cfg['base_opt']}, lr={best_cfg['lr']}, ρ={best_cfg['rho']}），"
                  f"最终损失 {final_losses[best_idx]:.6f}")
    lines.append(f"- 最差配置为 **Run {worst_idx+1}**（{worst_cfg['base_opt']}, lr={worst_cfg['lr']}, ρ={worst_cfg['rho']}），"
                  f"最终损失 {final_losses[worst_idx]:.6f}")
    if gap > 1.0:
        lines.append(f"- 最优与最差之间差距达 **{gap:.4f}**，参数选择对优化效果影响**极大**。")
    elif gap > 0.01:
        lines.append(f"- 最优与最差之间差距为 **{gap:.6f}**，参数选择对优化效果有**明显影响**。")
    else:
        lines.append(f"- 最优与最差之间差距仅 **{gap:.6f}**，各配置表现接近，参数敏感度较低。")

    base_opts_used = set(r['config']['base_opt'] for r in results)
    if len(base_opts_used) > 1:
        opt_avg = {}
        opt_cnt = {}
        for i, r in enumerate(results):
            bo = r['config']['base_opt']
            opt_avg[bo] = opt_avg.get(bo, 0) + final_losses[i]
            opt_cnt[bo] = opt_cnt.get(bo, 0) + 1
        opt_avg = {k: opt_avg[k] / opt_cnt[k] for k in opt_avg}
        best_opt = min(opt_avg, key=opt_avg.get)
        lines.append(f"\n**基础优化器对比：**")
        for bo, avg in sorted(opt_avg.items(), key=lambda x: x[1]):
            lines.append(f"- {bo}：平均最终损失 = {avg:.6f}")
        lines.append(f"> 💡 综合来看，**{best_opt}** 在本组实验中平均表现最佳。")

    rho_vals = [r['config']['rho'] for r in results]
    if max(rho_vals) - min(rho_vals) >= 0.1:
        lines.append(f"\n**ρ 的影响：**")
        low_rho = [(i, final_losses[i]) for i in range(n) if results[i]['config']['rho'] <= np.median(rho_vals)]
        high_rho = [(i, final_losses[i]) for i in range(n) if results[i]['config']['rho'] > np.median(rho_vals)]
        avg_low = np.mean([l for _, l in low_rho]) if low_rho else 0
        avg_high = np.mean([l for _, l in high_rho]) if high_rho else 0
        lines.append(f"- 较小 ρ（≤{np.median(rho_vals):.2f}）平均损失 = {avg_low:.6f}")
        lines.append(f"- 较大 ρ（>{np.median(rho_vals):.2f}）平均损失 = {avg_high:.6f}")
        if avg_low < avg_high:
            lines.append("> 💡 较小的 ρ 倾向于产生更稳定的优化结果（更小的扰动 = 更少的震荡）。")
        else:
            lines.append("> 💡 较大的 ρ 在本组实验中表现更好，说明适度的邻域探索有助于逃离局部极小。")

    lines.append(f"\n**收敛行为统计：**")
    phase_counts = {}
    for p in phases:
        phase_counts[p] = phase_counts.get(p, 0) + 1
    for p, cnt in sorted(phase_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {p}：{cnt} 个配置")

    eff_list = []
    for i, r in enumerate(results):
        straight = np.sqrt((r['pos'][-1, 0] - r['pos'][0, 0])**2 + (r['pos'][-1, 1] - r['pos'][0, 1])**2)
        eff = straight / total_dists[i] if total_dists[i] > 1e-10 else 0
        eff_list.append((i, eff))
    eff_list.sort(key=lambda x: -x[1])
    lines.append(f"\n**路径效率排名（越高越直接）：**")
    for i, (idx, eff) in enumerate(eff_list):
        cfg = results[idx]['config']
        bar = "█" * int(eff * 20) + "░" * (20 - int(eff * 20))
        lines.append(f"- Run {idx+1}：{bar} {eff:.1%} ({cfg['base_opt']}, ρ={cfg['rho']})")

    lines.append(f"\n**📋 总结建议：**")
    if gap > 1.0:
        lines.append(f"> 本组实验中参数选择至关重要。建议以 Run {best_idx+1} 的配置为基础，进一步微调学习率和 ρ 以追求更优结果。")
    elif gap > 0.01:
        lines.append(f"> 各配置存在一定差异，Run {best_idx+1} 表现最优。可尝试在其参数附近做更精细的网格搜索。")
    else:
        lines.append(f"> 各配置表现相近，说明当前测试函数和参数范围内优化器行为较为一致。建议尝试更极端的参数设置或更复杂的测试函数来区分优化器能力。")

    return "\n\n".join(lines)


# ==================== 绘制函数 ====================
def _auto_zoom(all_pos, margin=0.2):
    all_x, all_y = all_pos[:, 0], all_pos[:, 1]
    x_span = max(all_x.max() - all_x.min(), 0.5)
    y_span = max(all_y.max() - all_y.min(), 0.5)
    x_center = (all_x.min() + all_x.max()) / 2
    y_center = (all_y.min() + all_y.max()) / 2
    return ([x_center - x_span/2 - margin, x_center + x_span/2 + margin],
            [y_center - y_span/2 - margin, y_center + y_span/2 + margin])


def build_3d_fig(results, step_idx=None, resolution=GRID_RESOLUTION_HIGH):
    X, Y, x_range, y_range, Z = _get_grid_for_result(results[0], resolution)
    fig = go.Figure()
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.6, showscale=False))
    for i, res in enumerate(results):
        pos = res['pos']
        color = COLORS[i % len(COLORS)]
        cfg = res['config']
        label = f"Run {i+1}: {cfg['base_opt']}, ρ={cfg['rho']}, lr={cfg['lr']}"
        sl = slice(0, step_idx + 1) if step_idx is not None else slice(None)
        fig.add_trace(go.Scatter3d(
            x=pos[sl, 0], y=pos[sl, 1], z=res['loss'][sl],
            mode='lines+markers', marker=dict(size=3, color=color),
            line=dict(color=color, width=2), name=label
        ))
        fig.add_trace(go.Scatter3d(
            x=[pos[0, 0]], y=[pos[0, 1]], z=[res['loss'][0]],
            mode='markers', marker=dict(size=5, color=color, symbol='circle-open'), showlegend=False
        ))
        end = step_idx if step_idx is not None else len(pos) - 1
        fig.add_trace(go.Scatter3d(
            x=[pos[end, 0]], y=[pos[end, 1]], z=[res['loss'][end]],
            mode='markers', marker=dict(size=5, color=color, symbol='x'), showlegend=False
        ))
    fig.update_layout(height=550, scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='loss'),
                      margin=dict(l=0, r=0, t=30, b=0))
    return fig

def build_2d_fig(results, step_idx=None, resolution=GRID_RESOLUTION_HIGH):
    X, Y, x_range, y_range, Z = _get_grid_for_result(results[0], resolution)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("完整视图", "局部放大"))
    fig.add_trace(go.Contour(
        x=x_range, y=y_range, z=Z,
        colorscale='Viridis', opacity=0.5, showscale=False
    ), row=1, col=1)
    fig.add_trace(go.Contour(
        x=x_range, y=y_range, z=Z,
        colorscale='Viridis', opacity=0.5, showscale=False
    ), row=1, col=2)

    func_name = results[0]['config']['func']
    optima = TEST_FUNCTIONS[func_name].get('optima', [])
    for ox, oy in optima:
        for col in (1, 2):
            fig.add_trace(go.Scatter(
                x=[ox], y=[oy], mode='markers',
                marker=dict(size=12, color='gold', symbol='star',
                            line=dict(width=2, color='white')),
                name='理论最优', showlegend=(col == 1)
            ), row=1, col=col)

    for i, res in enumerate(results):
        pos = res['pos']
        color = COLORS[i % len(COLORS)]
        label = f"Run {i+1}"
        sl = slice(0, step_idx + 1) if step_idx is not None else slice(None)
        for col in (1, 2):
            fig.add_trace(go.Scatter(
                x=pos[sl, 0], y=pos[sl, 1], mode='lines+markers',
                marker=dict(size=4, color=color), line=dict(color=color, width=2),
                name=label if col == 1 else None, showlegend=(col == 1)
            ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=[pos[0, 0]], y=[pos[0, 1]], mode='markers',
            marker=dict(size=8, color=color, symbol='star', line=dict(width=1, color='white')),
            showlegend=False
        ), row=1, col=1)

        end = step_idx if step_idx is not None else len(pos) - 1
        rho_val = res['config']['rho']
        theta = np.linspace(0, 2 * np.pi, 60)
        circ_x = pos[end, 0] + rho_val * np.cos(theta)
        circ_y = pos[end, 1] + rho_val * np.sin(theta)
        for col in (1, 2):
            fig.add_trace(go.Scatter(
                x=circ_x, y=circ_y, mode='lines',
                line=dict(color=color, width=1, dash='dot'),
                opacity=0.5, showlegend=False,
                name=f'扰动范围 ρ={rho_val}' if col == 1 else None
            ), row=1, col=col)

    fig.update_xaxes(range=[x_range[0], x_range[-1]], row=1, col=1)
    fig.update_yaxes(range=[y_range[0], y_range[-1]], row=1, col=1)

    all_pos = np.concatenate([r['pos'] for r in results])
    xr, yr = _auto_zoom(all_pos)
    fig.update_xaxes(range=xr, row=1, col=2)
    fig.update_yaxes(range=yr, row=1, col=2)
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0))
    return fig

def build_conv_fig(results, step_idx=None, smooth=False):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("损失下降", "梯度范数", "扰动幅度"))
    for i, res in enumerate(results):
        color = COLORS[i % len(COLORS)]
        label = f"Run {i+1}"
        sl = slice(0, step_idx + 1) if step_idx is not None else slice(None)
        loss_data = res['loss'][sl]
        grad_data = res['grad_norm'][sl]
        perturb_data = res['perturb'][sl]
        if smooth:
            loss_display = _ema_smooth(loss_data)
            grad_display = _ema_smooth(grad_data)
            perturb_display = _ema_smooth(perturb_data)
        else:
            loss_display = loss_data
            grad_display = grad_data
            perturb_display = perturb_data
        x_axis = np.arange(len(loss_display))
        fig.add_trace(go.Scatter(
            x=x_axis, y=loss_display, mode='lines', name=label, line=dict(color=color)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x_axis, y=grad_display, mode='lines', showlegend=False, line=dict(color=color)
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=x_axis, y=perturb_display, mode='lines', showlegend=False, line=dict(color=color)
        ), row=1, col=3)
    fig.update_xaxes(title_text="迭代步", row=1, col=1)
    fig.update_xaxes(title_text="迭代步", row=1, col=2)
    fig.update_xaxes(title_text="迭代步", row=1, col=3)
    use_log_loss = all(np.all(r['loss'][r['loss'] > 0] > 0) for r in results) and all(len(r['loss']) > 0 for r in results)
    fig.update_yaxes(title_text="损失值", type="log" if use_log_loss else "linear", row=1, col=1)
    fig.update_yaxes(title_text="梯度范数", type="log", row=1, col=2)
    fig.update_yaxes(title_text="扰动大小", type="log", row=1, col=3)
    fig.update_layout(height=380, hovermode='x unified', margin=dict(l=0, r=0, t=30, b=0))
    return fig


def build_radar_fig(results):
    categories = ['收敛速度', '最终精度', '路径效率', '稳定性', '扰动利用', '最优接近']
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    fig = go.Figure()
    for i, res in enumerate(results):
        scores = _score_run(res)
        values = [scores[c] for c in categories]
        values += values[:1]
        color = COLORS[i % len(COLORS)]
        cfg = res['config']
        label = f"Run {i+1}: {cfg['base_opt']}, ρ={cfg['rho']}"
        fig.add_trace(go.Scatterpolar(
            r=values, theta=angles, mode='lines+markers',
            name=label, line=dict(color=color),
            marker=dict(size=5, color=color),
            fill='toself', opacity=0.3
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.0], tickvals=[0.2, 0.4, 0.6, 0.8, 1.0]),
            angularaxis=dict(tickvals=angles[:-1], ticktext=categories)
        ),
        height=450,
        margin=dict(l=30, r=30, t=40, b=30),
        title="🎯 优化器行为指纹"
    )
    return fig


def build_compare_trajectory_fig(prev_result, curr_result, step_idx, prev_step):
    func_name = curr_result['config']['func']
    _, X, Y, x_range, y_range, Z = get_test_grid(func_name, GRID_RESOLUTION_HIGH)
    fig_cmp = make_subplots(rows=1, cols=2, subplot_titles=("上一次 (虚线) vs 当前 (实线)", "局部放大"))
    fig_cmp.add_trace(go.Contour(
        x=x_range, y=y_range, z=Z,
        colorscale='Viridis', opacity=0.4, showscale=False
    ), row=1, col=1)
    fig_cmp.add_trace(go.Contour(
        x=x_range, y=y_range, z=Z,
        colorscale='Viridis', opacity=0.4, showscale=False
    ), row=1, col=2)

    optima = TEST_FUNCTIONS[func_name].get('optima', [])
    for ox, oy in optima:
        for col in (1, 2):
            fig_cmp.add_trace(go.Scatter(
                x=[ox], y=[oy], mode='markers',
                marker=dict(size=12, color='gold', symbol='star', line=dict(width=2, color='white')),
                name='理论最优', showlegend=(col == 1)
            ), row=1, col=col)

    prev_pos = prev_result['pos']
    curr_pos = curr_result['pos']
    for col in (1, 2):
        fig_cmp.add_trace(go.Scatter(
            x=prev_pos[:prev_step+1, 0], y=prev_pos[:prev_step+1, 1],
            mode='lines', name='上一次' if col == 1 else None,
            line=dict(color='gray', dash='dash', width=2),
            showlegend=(col == 1)
        ), row=1, col=col)
        fig_cmp.add_trace(go.Scatter(
            x=curr_pos[:step_idx+1, 0], y=curr_pos[:step_idx+1, 1],
            mode='lines+markers', name='当前' if col == 1 else None,
            marker=dict(size=4, color='red'), line=dict(color='red', width=2),
            showlegend=(col == 1)
        ), row=1, col=col)
    fig_cmp.add_trace(go.Scatter(
        x=[prev_pos[0, 0]], y=[prev_pos[0, 1]], mode='markers',
        marker=dict(size=8, color='gray', symbol='star', line=dict(width=1, color='white')),
        showlegend=False
    ), row=1, col=1)
    fig_cmp.add_trace(go.Scatter(
        x=[curr_pos[0, 0]], y=[curr_pos[0, 1]], mode='markers',
        marker=dict(size=8, color='red', symbol='star', line=dict(width=1, color='white')),
        showlegend=False
    ), row=1, col=1)
    fig_cmp.update_xaxes(range=[x_range[0], x_range[-1]], row=1, col=1)
    fig_cmp.update_yaxes(range=[y_range[0], y_range[-1]], row=1, col=1)
    all_px = np.concatenate([prev_pos[:prev_step+1, 0], curr_pos[:step_idx+1, 0]])
    all_py = np.concatenate([prev_pos[:prev_step+1, 1], curr_pos[:step_idx+1, 1]])
    zoom_pos = np.column_stack([all_px, all_py])
    xr, yr = _auto_zoom(zoom_pos, margin=0.3)
    fig_cmp.update_xaxes(range=xr, row=1, col=2)
    fig_cmp.update_yaxes(range=yr, row=1, col=2)
    fig_cmp.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0))
    return fig_cmp

def build_compare_loss_fig(prev_result, curr_result):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(prev_result['loss'])), y=prev_result['loss'],
        mode='lines', name='上一次', line=dict(color='gray', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(curr_result['loss'])), y=curr_result['loss'],
        mode='lines', name='当前', line=dict(color='red')
    ))
    use_log = np.all(prev_result['loss'] > 0) and np.all(curr_result['loss'] > 0)
    fig.update_layout(xaxis_title="迭代步", yaxis_title="损失值",
                      yaxis_type="log" if use_log else "linear",
                      height=350, hovermode='x unified',
                      margin=dict(l=0, r=0, t=30, b=0))
    return fig


def render_config_badge(config, prefix="📌"):
    sched_parts = []
    lr_s = config.get('lr_scheduler', 'none')
    rho_s = config.get('rho_scheduler', 'none')
    if lr_s != 'none':
        sched_parts.append(f"lr调度={lr_s}")
    if rho_s != 'none':
        sched_parts.append(f"ρ调度={rho_s}")
    gc = config.get('grad_clip', 0.0)
    if gc > 0:
        sched_parts.append(f"裁剪={gc}")
    sched_str = f" | {' | '.join(sched_parts)}" if sched_parts else ""

    st.caption(
        f"{prefix} **{config['func']}** | {config['base_opt']} | lr={config['lr']} | ρ={config['rho']} | "
        f"自适应={'开' if config['adaptive'] else '关'} | K={config['update_freq']} | "
        f"损失={config['loss_type']} | 起点=({config['init_x']}, {config['init_y']}){sched_str}"
    )

def render_chart(results, step_idx, resolution=GRID_RESOLUTION_HIGH, smooth=False):
    tab_3d, tab_2d, tab_conv = st.tabs(["🌄 3D 表面", "🗺️ 2D 等高线", "📉 收敛曲线"])
    with tab_3d:
        st.plotly_chart(build_3d_fig(results, step_idx=step_idx, resolution=resolution), use_container_width=True)
    with tab_2d:
        st.plotly_chart(build_2d_fig(results, step_idx=step_idx, resolution=resolution), use_container_width=True)
    with tab_conv:
        st.plotly_chart(build_conv_fig(results, step_idx=step_idx, smooth=smooth), use_container_width=True)

def render_chart_fast(chart_type, results, step_idx, smooth=False):
    builder = {"🌄 3D 表面": lambda r, si: build_3d_fig(r, si, resolution=GRID_RESOLUTION_LOW),
               "🗺️ 2D 等高线": lambda r, si: build_2d_fig(r, si, resolution=GRID_RESOLUTION_LOW)}.get(chart_type,
               lambda r, si: build_conv_fig(r, si, smooth=smooth))
    st.plotly_chart(builder(results, step_idx), use_container_width=True)

st.set_page_config(page_title="BiSAM 多维可视化", layout="wide")
st.title("🔬 BiSAM 优化器 — 多维对比分析")

for key, default in [('compare_configs', []), ('compare_results', []), ('preview_history', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==================== 自动播放：在 widget 创建之前递增步数 ====================
if '_preview_auto_advance' not in st.session_state:
    st.session_state._preview_auto_advance = False
if '_compare_auto_advance' not in st.session_state:
    st.session_state._compare_auto_advance = False

if st.session_state._preview_auto_advance:
    _pmax = st.session_state.get('_preview_max_step', 0)
    _pcur = st.session_state.get('preview_step', 0)
    if _pcur < _pmax:
        st.session_state.preview_step = _pcur + 1
    else:
        st.session_state._preview_auto_advance = False

if st.session_state._compare_auto_advance:
    _cmax = st.session_state.get('_compare_max_step', 0)
    _ccur = st.session_state.get('compare_step', 0)
    if _ccur < _cmax:
        st.session_state.compare_step = _ccur + 1
    else:
        st.session_state._compare_auto_advance = False

def _on_random_init():
    info = TEST_FUNCTIONS[st.session_state.func]
    st.session_state._pending_random_init = {
        'init_x': round(np.random.uniform(*info['x_range']), 2),
        'init_y': round(np.random.uniform(*info['y_range']), 2),
    }

def _on_load_preset():
    func_name = st.session_state.func
    if func_name in PRESET_CONFIGS:
        st.session_state._pending_preset = PRESET_CONFIGS[func_name].copy()

if '_pending_random_init' in st.session_state and st.session_state._pending_random_init:
    _ri = st.session_state._pending_random_init
    st.session_state.init_x = _ri['init_x']
    st.session_state.init_y = _ri['init_y']
    del st.session_state._pending_random_init
    st.rerun()
if '_pending_preset' in st.session_state and st.session_state._pending_preset:
    _preset = st.session_state._pending_preset
    for _k, _v in _preset.items():
        if _k != 'func':
            st.session_state[_k] = _v
    del st.session_state._pending_preset
    st.rerun()

mode = st.radio("选择模式", ["⚡ 即时预览", "📊 多配置对比"], horizontal=True)

# ==================== 侧边栏参数 ====================
with st.sidebar:
    st.header("🎛️ 参数配置")
    func = st.selectbox("测试函数", list(TEST_FUNCTIONS.keys()), key="func")
    base_opt = st.selectbox("基础优化器", ["SGD", "Adam", "AdamW"], key="base_opt")
    default_lr = DEFAULT_LR.get(base_opt, 0.05)
    if 'last_base_opt' not in st.session_state:
        st.session_state.last_base_opt = base_opt
    if st.session_state.last_base_opt != base_opt:
        st.session_state.lr = default_lr
        st.session_state.last_base_opt = base_opt
    if 'lr' not in st.session_state:
        st.session_state.lr = default_lr
    lr = st.number_input("学习率", step=0.005, format="%.4f", min_value=1e-6, key="lr")
    rho = st.slider("ρ (扰动半径)", 0.05, 2.0, 0.5, 0.05, key="rho")
    adaptive = st.checkbox("自适应 (ASAM)", True, key="adaptive")
    update_freq = st.slider("更新频率 K", 1, 10, 5, key="update_freq")
    loss_type = st.selectbox("下界损失", ["tanh", "log"], key="loss_type")
    if loss_type == "tanh":
        alpha = st.slider("α (tanh)", 0.01, 0.5, 0.1)
        mu = 1.0
    else:
        alpha = 0.1
        mu = st.slider("μ (log)", 0.1, 3.0, 1.0)

    st.markdown("---")
    st.markdown("**⏱️ 调度策略**")
    lr_scheduler = st.selectbox("学习率调度", ["none", "cosine", "linear", "warmup_cosine"],
                                format_func=lambda x: {"none": "无", "cosine": "余弦退火", "linear": "线性衰减", "warmup_cosine": "Warmup+余弦"}[x],
                                key="lr_scheduler")
    rho_scheduler = st.selectbox("ρ 调度", ["none", "cosine", "linear"],
                                 format_func=lambda x: {"none": "无", "cosine": "余弦退火", "linear": "线性衰减"}[x],
                                 key="rho_scheduler")

    st.markdown("---")
    st.markdown("**🛡️ 梯度裁剪**")
    grad_clip = st.slider("最大梯度范数 (0=不裁剪)", 0.0, 10.0, 0.0, 0.5, key="grad_clip")

    steps = st.slider("优化步数", 10, 300, 50, key="steps")
    st.markdown("---")
    st.markdown("**📍 起始位置**")
    init_x = st.number_input("初始 x", value=2.5, key="init_x")
    init_y = st.number_input("初始 y", value=2.5, key="init_y")
    if st.button("🎲 随机初始位置", key="random_init", on_click=_on_random_init):
        pass
    st.markdown("---")
    if func in PRESET_CONFIGS:
        if st.button(f"📋 加载 {func} 推荐配置", key="load_preset", use_container_width=True, on_click=_on_load_preset):
            pass

    current_config = {
        'base_opt': base_opt, 'lr': lr, 'rho': rho,
        'adaptive': adaptive, 'update_freq': update_freq,
        'loss_type': loss_type, 'alpha': alpha, 'mu': mu,
        'func': func, 'init_x': init_x, 'init_y': init_y,
        'lr_scheduler': lr_scheduler, 'rho_scheduler': rho_scheduler,
        'grad_clip': grad_clip
    }

# ==================== 即时预览模式 ====================
if mode == "⚡ 即时预览":
    run_key = f"steps={steps}_" + "_".join(
        f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in current_config.items()
    )

    if 'last_run_key' not in st.session_state or st.session_state.last_run_key != run_key:
        if 'preview_result' in st.session_state and 'last_run_key' in st.session_state:
            prev = {
                'result': copy.deepcopy(st.session_state.preview_result),
                'config': copy.deepcopy(st.session_state.get('preview_config', current_config)),
                'key': st.session_state.last_run_key
            }
            st.session_state.preview_history.append(prev)
            if len(st.session_state.preview_history) > MAX_PREVIEW_HISTORY:
                st.session_state.preview_history = st.session_state.preview_history[-MAX_PREVIEW_HISTORY:]

        with st.spinner("正在计算..."):
            try:
                st.session_state.preview_result = run_optimization(current_config, steps=steps)
                st.session_state.preview_config = current_config.copy()
                st.session_state.last_run_key = run_key
            except Exception as e:
                st.error(f"计算出错: {e}")
                st.stop()

    result = st.session_state.preview_result
    max_step = len(result['pos']) - 1

    st.markdown("#### 📌 当前参数")
    render_config_badge(current_config)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("最终损失", f"{result['loss'][-1]:.6f}",
              delta=f"{result['loss'][-1] - result['loss'][0]:.6f}", delta_color="inverse")
    c2.metric("最低损失", f"{result['loss'].min():.6f}")
    c3.metric("终点位置", f"({result['pos'][-1, 0]:.3f}, {result['pos'][-1, 1]:.3f})")
    c4.metric("总步数", f"{steps}")

    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns([4, 2, 2, 2, 2])
    with ctrl_col1:
        step_idx = st.slider("🔍 拖动查看逐步优化过程", 0, max_step, max_step, key="preview_step")
    with ctrl_col2:
        auto_play = st.checkbox("▶ 自动播放", key="auto_play")
    with ctrl_col3:
        play_speed = st.select_slider("速度", options=[0.5, 1, 2, 4, 8], value=1,
                                      format_func=lambda x: f"{x}x", key="play_speed")
    with ctrl_col4:
        smooth_toggle = st.checkbox("📈 EMA平滑", key="smooth_preview", value=False)
    with ctrl_col5:
        if st.button("⏹ 停止", key="stop_play", disabled=not auto_play):
            st.session_state._preview_auto_advance = False
            st.session_state.preview_step = max_step
            st.rerun()

    st.info(f"**第 {step_idx}/{max_step} 步** | "
            f"位置: ({result['pos'][step_idx, 0]:.4f}, {result['pos'][step_idx, 1]:.4f}) | "
            f"损失: {result['loss'][step_idx]:.6f}")

    is_playing = auto_play and step_idx < max_step
    if is_playing:
        chart_view = st.selectbox(
            "播放视图", ["🗺️ 2D 等高线", "🌄 3D 表面", "📉 收敛曲线"],
            key="play_chart_view", label_visibility="collapsed"
        )
        render_chart_fast(chart_view, [result], step_idx=step_idx, smooth=smooth_toggle)
    else:
        render_chart([result], step_idx=step_idx, smooth=smooth_toggle)

    if not is_playing:
        with st.expander("📝 优化过程详细分析", expanded=True):
            st.markdown(generate_single_analysis(result))

    # ---- 前后对比区域 ----
    if len(st.session_state.preview_history) > 0:
        st.markdown("---")
        hist_header, hist_clear = st.columns([4, 1])
        with hist_header:
            st.markdown(f"#### 🔄 前后对比（当前 vs 上一次，共 {len(st.session_state.preview_history)} 条历史）")
        with hist_clear:
            if st.button("🗑️ 清除历史", key="clear_history"):
                st.session_state.preview_history.clear()
                st.rerun()

        last_hist = st.session_state.preview_history[-1]
        prev_result = last_hist['result']
        prev_config = last_hist['config']
        prev_max_step = len(prev_result['pos']) - 1
        prev_step = min(step_idx, prev_max_step)

        st.markdown("**上一次配置:**")
        render_config_badge(prev_config, prefix="📎")

        cmp_tab1, cmp_tab2 = st.tabs(["🗺️ 轨迹对比", "📉 损失对比"])
        with cmp_tab1:
            st.plotly_chart(
                build_compare_trajectory_fig(prev_result, result, step_idx, prev_step),
                use_container_width=True
            )
        with cmp_tab2:
            st.plotly_chart(
                build_compare_loss_fig(prev_result, result),
                use_container_width=True
            )

        mc1, mc2 = st.columns(2)
        mc1.metric("最终损失变化", f"{result['loss'][-1]:.6f}",
                   delta=f"{result['loss'][-1] - prev_result['loss'][-1]:.6f}", delta_color="inverse")
        mc2.metric("最低损失变化", f"{result['loss'].min():.6f}",
                   delta=f"{result['loss'].min() - prev_result['loss'].min():.6f}", delta_color="inverse")

        with st.expander("🔍 参数变化影响分析", expanded=True):
            st.markdown(generate_diff_analysis(prev_result, prev_config, result, current_config))

    st.markdown("---")
    if st.button("💾 将当前配置保存到对比列表", use_container_width=True):
        st.session_state.compare_configs.append(current_config.copy())
        st.success(f"已保存！对比列表现有 {len(st.session_state.compare_configs)} 个配置")

    if is_playing:
        st.session_state._preview_max_step = max_step
        st.session_state._preview_auto_advance = True
        time.sleep(AUTO_PLAY_BASE_INTERVAL / play_speed)
        st.rerun()
    elif auto_play and step_idx >= max_step:
        st.session_state._preview_auto_advance = False

# ==================== 多配置对比模式 ====================
else:
    with st.sidebar:
        st.markdown("---")
        col_add, col_clear = st.columns(2)
        with col_add:
            if st.button("➕ 添加配置"):
                st.session_state.compare_configs.append(current_config.copy())
                st.success(f"已添加 #{len(st.session_state.compare_configs)}")
        with col_clear:
            if st.button("🗑️ 清空"):
                st.session_state.compare_configs.clear()
                st.session_state.compare_results.clear()
                st.rerun()

        if len(st.session_state.compare_configs) > 0:
            st.info(f"已保存 {len(st.session_state.compare_configs)} 个配置")
            cfg_labels = [
                f"#{i+1}: {c['base_opt']}, ρ={c['rho']}, lr={c['lr']}, {c['func']}"
                for i, c in enumerate(st.session_state.compare_configs)
            ]
            selected_idx = st.selectbox("选择要删除的配置", range(len(cfg_labels)),
                                        format_func=lambda x: cfg_labels[x], key="del_select")
            if st.button("🗑️ 删除选中"):
                st.session_state.compare_configs.pop(selected_idx)
                st.session_state.compare_results.clear()
                st.rerun()

        st.markdown("---")
        st.markdown("**📁 配置导入/导出**")
        if st.button("📤 导出配置为 JSON", key="export_configs", use_container_width=True):
            st.session_state._export_json = json.dumps(st.session_state.compare_configs, indent=2, ensure_ascii=False)

        if '_export_json' in st.session_state and st.session_state._export_json:
            st.code(st.session_state._export_json, language="json")

        import_json = st.text_area("📥 粘贴 JSON 导入配置", key="import_json_area", height=80)
        if st.button("📥 导入配置", key="import_configs", use_container_width=True):
            try:
                imported = json.loads(import_json)
                if isinstance(imported, list):
                    st.session_state.compare_configs.extend(imported)
                    st.success(f"成功导入 {len(imported)} 个配置！")
                    st.rerun()
                else:
                    st.error("JSON 格式错误：需要一个列表")
            except json.JSONDecodeError as e:
                st.error(f"JSON 解析失败: {e}")

        if st.button("▶️ 开始对比优化", type="primary"):
            results = []
            errors = []
            prog = st.progress(0)
            with st.spinner(f"正在运行 {len(st.session_state.compare_configs)} 个配置..."):
                for idx, cfg in enumerate(st.session_state.compare_configs):
                    try:
                        res = run_optimization(cfg, steps=steps)
                        results.append(res)
                    except Exception as e:
                        errors.append((idx + 1, str(e)))
                    prog.progress((idx + 1) / len(st.session_state.compare_configs))
            st.session_state.compare_results = results
            if errors:
                for run_id, err_msg in errors:
                    st.error(f"Run {run_id} 失败: {err_msg}")
            if results:
                st.success(f"完成！成功 {len(results)}/{len(st.session_state.compare_configs)}")
            else:
                st.error("所有配置均失败，请检查参数设置")

    if len(st.session_state.compare_results) > 0:
        results = st.session_state.compare_results
        r0 = results[0]
        all_same_func = all(r['config']['func'] == r0['config']['func'] for r in results)

        if not all_same_func:
            st.warning("⚠️ 不同配置使用了不同测试函数，表面/等高线仅展示第一个配置的函数。")

        min_steps = min(len(r['pos']) for r in results) - 1
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5 = st.columns([4, 2, 2, 2, 2])
        with ctrl_col1:
            step_idx = st.slider("🔍 拖动查看逐步优化过程", 0, min_steps, min_steps, key="compare_step")
        with ctrl_col2:
            compare_auto_play = st.checkbox("▶ 自动播放", key="compare_auto_play")
        with ctrl_col3:
            cmp_play_speed = st.select_slider("速度", options=[0.5, 1, 2, 4, 8], value=1,
                                              format_func=lambda x: f"{x}x", key="cmp_play_speed")
        with ctrl_col4:
            cmp_smooth = st.checkbox("📈 EMA平滑", key="smooth_compare", value=False)
        with ctrl_col5:
            if st.button("⏹ 停止", key="stop_cmp_play", disabled=not compare_auto_play):
                st.session_state._compare_auto_advance = False
                st.session_state.compare_step = min_steps
                st.rerun()

        with st.expander("📌 各配置参数详情", expanded=False):
            param_table = []
            for i, res in enumerate(results):
                cfg = res['config']
                param_table.append({
                    "": f"Run {i+1}",
                    "函数": cfg['func'],
                    "优化器": cfg['base_opt'],
                    "lr": cfg['lr'],
                    "ρ": cfg['rho'],
                    "自适应": "✓" if cfg['adaptive'] else "✗",
                    "K": cfg['update_freq'],
                    "损失类型": cfg['loss_type'],
                    "lr调度": cfg.get('lr_scheduler', 'none'),
                    "ρ调度": cfg.get('rho_scheduler', 'none'),
                    "梯度裁剪": cfg.get('grad_clip', 0.0),
                    "起点": f"({cfg['init_x']}, {cfg['init_y']})"
                })
            st.dataframe(param_table, use_container_width=True)

        st.markdown("#### 📊 关键指标对比")
        n_results = len(results)
        for row_start in range(0, n_results, 5):
            row_end = min(row_start + 5, n_results)
            metric_cols = st.columns(row_end - row_start)
            for j, i in enumerate(range(row_start, row_end)):
                res = results[i]
                with metric_cols[j]:
                    st.metric(f"Run {i+1}", f"{res['loss'][-1]:.6f}",
                              delta=f"Δ{res['loss'][-1] - res['loss'][0]:.4f}", delta_color="inverse")

        is_cmp_playing = compare_auto_play and step_idx < min_steps
        if is_cmp_playing:
            cmp_chart_view = st.selectbox(
                "播放视图", ["🗺️ 2D 等高线", "🌄 3D 表面", "📉 收敛曲线"],
                key="cmp_play_chart_view", label_visibility="collapsed"
            )
            render_chart_fast(cmp_chart_view, results, step_idx=step_idx, smooth=cmp_smooth)
        else:
            render_chart(results, step_idx=step_idx, smooth=cmp_smooth)

        st.markdown("#### 📋 详细数据")
        table_data = []
        for i, res in enumerate(results):
            cfg = res['config']
            table_data.append({
                "运行": f"Run {i+1}",
                "优化器": cfg['base_opt'],
                "学习率": cfg['lr'],
                "ρ": cfg['rho'],
                "自适应": "✓" if cfg['adaptive'] else "✗",
                "K": cfg['update_freq'],
                "损失类型": cfg['loss_type'],
                "函数": cfg['func'],
                "起点": f"({cfg['init_x']}, {cfg['init_y']})",
                "终点": f"({res['pos'][-1, 0]:.3f}, {res['pos'][-1, 1]:.3f})",
                "最终损失": f"{res['loss'][-1]:.6f}",
                "最低损失": f"{res['loss'].min():.6f}",
            })
        st.dataframe(table_data, use_container_width=True)

        if not is_cmp_playing:
            st.markdown("#### 🎯 优化器行为指纹")
            st.plotly_chart(build_radar_fig(results), use_container_width=True)

            with st.expander("🏆 综合分析与排名", expanded=True):
                st.markdown(generate_compare_analysis(results))

            for i, res in enumerate(results):
                with st.expander(f"📝 Run {i+1} 详细分析"):
                    st.markdown(generate_single_analysis(res))

        if is_cmp_playing:
            st.session_state._compare_max_step = min_steps
            st.session_state._compare_auto_advance = True
            time.sleep(AUTO_PLAY_BASE_INTERVAL / cmp_play_speed)
            st.rerun()
        elif compare_auto_play and step_idx >= min_steps:
            st.session_state._compare_auto_advance = False
    else:
        st.info("👈 请先在左侧配置参数，点击「添加配置」保存，然后点击「开始对比优化」。")

st.markdown("---")
st.caption("💡 即时预览：调参即刷新，自动保存历史可前后对比 | 多配置对比：保存多个配置一次性对比 | 支持 JSON 导入/导出配置")
