"""Feature-tokenizer Transformer with gated attention for tabular data (scikit-learn interface).

Each feature is embedded as a token (value x learned vector + bias); a [CLS] token is prepended.
Blocks: pre-LayerNorm -> multi-head self-attention whose output is multiplied by a
head-wise sigmoid gate computed from the block input (gated attention) -> residual;
pre-LayerNorm -> SwiGLU feed-forward -> residual. Prediction from the [CLS] token.
"""

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin


class GatedMHA(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.gate = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)
        self.last_attn = None
        self.last_gate = None

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        a = torch.softmax(q @ k.transpose(-2, -1) / self.dk**0.5, dim=-1)
        self.last_attn = a.detach()
        o = (self.drop(a) @ v).transpose(1, 2).reshape(B, N, D)
        g = torch.sigmoid(self.gate(x))
        self.last_gate = g.detach()
        return self.out(o * g)


class SwiGLU(nn.Module):
    def __init__(self, d, mult, drop):
        super().__init__()
        h = int(d * mult)
        self.w = nn.Linear(d, 2 * h)
        self.o = nn.Linear(h, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        a, b = self.w(x).chunk(2, -1)
        return self.o(self.drop(F.silu(a) * b))


class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att, self.ff = GatedMHA(d, h, drop), SwiGLU(d, 4 / 3, drop)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x + self.drop(self.att(self.n1(x)))
        return x + self.drop(self.ff(self.n2(x)))


class Net(nn.Module):
    def __init__(self, p, d=32, h=4, L=2, drop=0.15):
        super().__init__()
        self.W = nn.Parameter(torch.randn(p, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(p, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([Block(d, h, drop) for _ in range(L)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x):
        t = x.unsqueeze(-1) * self.W + self.b
        t = torch.cat([self.cls.expand(len(x), -1, -1), t], 1)
        for b in self.blocks:
            t = b(t)
        return self.head(t[:, 0]).squeeze(-1)


class _Base(BaseEstimator):
    def __init__(
        self,
        d=32,
        heads=4,
        layers=2,
        dropout=0.15,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=120,
        batch=256,
        patience=15,
        val_frac=0.15,
        random_state=0,
    ):
        self.d, self.heads, self.layers, self.dropout, self.lr, self.weight_decay = (
            d,
            heads,
            layers,
            dropout,
            lr,
            weight_decay,
        )
        self.epochs, self.batch, self.patience, self.val_frac, self.random_state = (
            epochs,
            batch,
            patience,
            val_frac,
            random_state,
        )

    def _prep(self, X, fit=False):
        X = np.asarray(X, dtype=np.float32)
        if fit:
            self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6
        return torch.tensor((X - self.mu_) / self.sd_)

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        torch.set_num_threads(2)
        Xt = self._prep(X, True)
        y = np.asarray(y, dtype=np.float32)
        if not self._binary:
            self.ym_, self.ys_ = y.mean(), y.std() + 1e-6
            y = (y - self.ym_) / self.ys_
        yt = torch.tensor(y)
        n = len(Xt)
        idx = np.random.permutation(n)
        nv = int(n * self.val_frac)
        va, tr = idx[:nv], idx[nv:]
        self.net_ = Net(Xt.shape[1], self.d, self.heads, self.layers, self.dropout)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lossf = nn.BCEWithLogitsLoss() if self._binary else nn.MSELoss()
        best, bad, state = np.inf, 0, None
        for ep in range(self.epochs):
            self.net_.train()
            perm = np.random.permutation(tr)
            for i in range(0, len(perm), self.batch):
                bi = perm[i : i + self.batch]
                opt.zero_grad()
                loss = lossf(self.net_(Xt[bi]), yt[bi])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
            self.net_.eval()
            with torch.no_grad():
                vl = lossf(self.net_(Xt[va]), yt[va]).item()
            if vl < best - 1e-4:
                best, bad, state = vl, 0, {k: v.clone() for k, v in self.net_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net_.load_state_dict(state)
        self.n_epochs_ = ep + 1
        return self

    def _raw(self, X):
        self.net_.eval()
        with torch.no_grad():
            return self.net_(self._prep(X)).numpy()

    def gate_values(self, X):
        """mean sigmoid gate value per token (CLS first) and layer: array (layers, tokens)"""
        self.net_.eval()
        with torch.no_grad():
            self.net_(self._prep(X))
            return np.stack([b.att.last_gate.mean((0, 2)).numpy() for b in self.net_.blocks])

    def cls_attention(self, X):
        """mean attention from [CLS] to each feature, averaged over heads, layers and samples"""
        self.net_.eval()
        out = []
        with torch.no_grad():
            self.net_(self._prep(X))
            for b in self.net_.blocks:
                out.append(b.att.last_attn[:, :, 0, 1:].mean((0, 1)).numpy())
        return np.mean(out, 0)


class GatedFTTRegressor(_Base, RegressorMixin):
    _binary = False

    def predict(self, X):
        return self._raw(X) * self.ys_ + self.ym_


class GatedFTTClassifier(_Base, ClassifierMixin):
    _binary = True

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return super().fit(X, y)

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self._raw(X)))
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
