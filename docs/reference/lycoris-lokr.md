I now have a complete picture of the code, docs, tests, and history. Here is the report.

---

# LyCORIS (v4.0.x, HEAD `4a6a333`) — LoKr-centric implementation report

**Repo:** `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS` (not a git worktree of yours; `git describe` = `v4.0.1.dev20260905071546`, `pyproject.toml` version 4.0.0).
**Scope note:** the repository is considerably larger than the prompt assumed — since 4.0.0 (2026-09-01) it ships a fused Triton/TileLang kernel subsystem (`lycoris/kernels/`, ~60 files) with per-call backend dispatch. There is no `lycoris/modules/attention.py`, no `glokr.py`, no `dora.py`; DoRA is a flag (`weight_decompose`) on locon/loha/lokr. No torch is installed on this machine, so all numeric claims below come from static reading plus running the pure-Python `factorization()` and a faithful re-implementation of the LoKr shape-selection logic.

---

## 1. `lycoris/modules/base.py` — `LycorisBaseModule`

### 1.1 `ModuleCustomSD` (lines 52–102)
Base of everything. Registers a load pre-hook (`load_weight_prehook`, line 58) and post-hook (`load_weight_hook`, line 70). Overrides `state_dict()` (lines 76–102): if the subclass' `custom_state_dict()` returns a dict, **that dict replaces** the default parameter dump (keys become `f"{prefix}{k}"`). This is how LoKr saves `alpha` as a buffer, folds `scalar` into `lokr_w1`, and omits non-persistent things. Loading still uses normal `nn.Module.load_state_dict` semantics against real parameter/buffer names, so saved key names must equal attribute names (they do for LoKr/LoHa/LoCon; `FullModule` remaps `diff`→`weight` in its pre-hook).

### 1.2 Constructor (lines 125–266)
```
LycorisBaseModule(lora_name, org_module, multiplier=1.0, dropout=0.0, rank_dropout=0.0,
                  module_dropout=0.0, rank_dropout_scale=False, bypass_mode=None, **kwargs)
```
- Lines 142–149: if `org_module` is a PEFT `BaseTunerLayer`, unwraps to its `base_layer`.
- Lines 151–222 detect the wrapped layer type and set `module_type`, `shape`, `op`, `dim`, `kw_dict`:
  - `nn.Linear` or a weight-only `Fp8Linear` duck-type (`is_weight_only_fp8_linear`, lines 17–24: class name `Fp8Linear` with `weight` + `weight_scale`) → `shape=(out,in)`, `op=F.linear`.
  - `Conv1d/2d/3d` → `shape=(out, in//groups, *k)`, `op=F.convNd`, `kw_dict={stride,padding,dilation,groups}`.
  - `LayerNorm` → `shape=normalized_shape`, `op=F.layer_norm`, `kw_dict={normalized_shape,eps}`; `GroupNorm` → `shape=(C,)`, `op=F.group_norm`.
  - anything else → `not_supported=True`, `module_type="unknown"` (apply_to/restore/merge become no-ops).
- Line 224: non-persistent `dtype_tensor` buffer used for the `dtype`/`device` properties (lines 317–323).
- **Automatic bypass selection (lines 226–246):**
  - weight-only FP8 Linear → `is_quant=True`, `bypass_mode=True` (warn via `log_fp8_bypass` if not already requested).
  - `isinstance(org_module, QuantLinears)` (bnb `Linear8bitLt/LinearFP4/LinearNF4`, quanto `QLinear/QConv2d/QLayerNorm`, optimum-quanto equivalents — `lycoris/utils/quant.py` lines 61–71; stub subclasses are defined when the libs are missing so the isinstance check is always safe) → force bypass.
  - Any linear-like whose class name is not exactly `"Linear"` (e.g. torchao, custom subclasses): if `bypass_mode is None` → `log_suspect()` and force bypass; if explicitly `True` → `is_quant=True`. Explicit `False` is respected.
- Lines 252–262 build `self.drop = nn.Dropout(dropout)` and `self.rank_drop = nn.Dropout(rank_dropout)` (or Identity). The comment block (253–258) documents the intended semantics: LoCon drops on the rank axis between A and B; every other algo drops rows of ΔW (rebuild) or drops ΔW·x (bypass).
- Line 264–266: `multiplier`, `org_forward = org_module.forward`, and `org_module = [org_module]` (list so the base weights are not registered as sub-parameters).

There is **no DoRA / `wd` / `rs_lora` / `use_scalar` in the base class** — those live in each algorithm module (see §2.6 for LoKr). `wd`, `dora_scale`, `scale`, `alpha`, `scalar` are per-module attributes.

### 1.3 `apply_to` / `restore` — forward replacement with stacking (lines 342–395)
Not a forward hook: `apply_to` sets `module.forward = self.forward` (instance attribute override) after saving `module._lycoris_original_forward` once and appending itself to `module._lycoris_wrappers`. `self.org_forward` is set to the *current* `module.forward` (which may be a previous wrapper's forward), so wrappers chain. `restore` pops itself out of the list, re-links the next wrapper's `org_forward` (line 385), and either reinstalls the previous top wrapper's forward or the original; when the list empties it deletes the two bookkeeping attributes. `FullModule` (full.py 109–127) and `IA3Module` (ia3.py 102–104) override `apply_to` and do **not** participate in stacking.

### 1.4 Merge API (lines 397–460, 480–657)
- `merge_to(multiplier=1.0, *, precise=False)` (397–418): raises for FP8 base; builds a `_MergeContext` (moves adapter params to the base weight's device and, if `precise`, to fp64), calls `get_merged_weight(multiplier, shape, device)` and copies into `weight.data` (and bias if returned). `precise=True` keeps fp64 CPU snapshots on the org module (`_lycoris_precise_weight_base/_current`, 609–657) so repeated merge/unmerge cycles do not drift (added 3.4.0; tested by `test/precision_merge_test.py`).
- `onfly_merge` / `onfly_restore` (420–454): caches the original weight on CPU, writes the merged weight, and restores later. Inference-time helper.
- Abstract: `get_diff_weight(multiplier, shape, device) -> (ΔW, Δb|None)` (456), `get_merged_weight` (459), `apply_max_norm` (462, default no-op), `bypass_forward_diff` (466), `bypass_forward` (469), `forward` (477).
- `parametrize_forward` (472–475): returns `get_merged_weight(multiplier, shape=x.shape)[0].to(x.dtype)` — used by the parametrize API.
- `_current_weight()` (333–336) returns the detached base weight, dequantizing FP8 via `dequantize_weight_only_fp8` (44–49).

### 1.5 `parametrize` classmethod (lines 268–303)
Builds a proxy `nn.Linear`/`nn.ConvNd` whose `.weight` *is* the target parameter, constructs `cls("", proxy, *args, bypass_mode=False, **kwargs)`, swaps `forward` for `parametrize_forward`, and calls `torch.nn.utils.parametrize.register_parametrization(org_module, attr, module_obj)`. `FullModule` is refused (272–273). **Bug:** line 277–279 `nn.Linear(target_param.shape[0], target_param.shape[1])` passes `(out, in)` as `(in_features, out_features)`, so for non-square weights every subclass reads swapped `in_dim/out_dim`; tests only use square layers.

### 1.6 Factory helpers
- `algo_check(state_dict, lora_name)` (305–307): any of `weight_list_det` keys present.
- `extract_state_dict` (309–311): returns tensors in `weight_list` order (None for absent).
- `make_module_from_state_dict` (313–315): abstract; implemented per algo.
- `make_module` / `get_module` are in `lycoris/modules/__init__.py` lines 34–47 (`MODULE_LIST` order at 19–31 decides detection priority: LoCon, LoHa, IA3, LoKr, Full, Norm, DiagOFT, BOFT, GLoRA, DyLoRA, TLoRA).

---

## 2. `lycoris/modules/lokr.py` — `LokrModule`

### 2.1 Math
For a Linear `W ∈ ℝ^{out×in}`:
- `factorization(out, factor) = (a, b)` with `a·b = out`, `a ≤ b`; `factorization(in, factor) = (c, d)` with `c·d = in`, `c ≤ d`.
- `ΔW = scale · (W1 ⊗ W2)`, `W1 ∈ ℝ^{a×c}` (small, "weight scale"), `W2 ∈ ℝ^{b×d}` (large, "weight"). `W1 ⊗ W2` is `(a·b) × (c·d) = out × in`. (Comment at line 151: `out_dim = a*c, in_dim = b*d` uses the transposed labeling; the code's actual shapes are `lokr_w1: (out_l, in_m)`, `lokr_w2: (out_k, in_n)`.)
- Rank: `rank(ΔW) ≤ rank(W1)·rank(W2)`; when W2 is low-rank `W2 = W2a·W2b` with rank r, `rank(ΔW) ≤ min(a,c)·r` (docs/algorithms/details.md 105–135).
- Conv: `W2 ∈ ℝ^{b×d×k1×k2}`, `W1` is unsqueezed to `(a,c,1,1)` and `torch.kron` broadcasts spatial dims (`make_kron`, functional/lokr.py 16–25).
- Bypass identity used to avoid ΔW: `(W1 ⊗ W2)·vec(X) = vec(W1 · X · W2ᵀ)` with `X = unvec(x)` as a `(c_in-blocks) × d` matrix — see §2.5.

### 2.2 `factor` and the shape decision (lines 82–174, verbatim below)
- `factor=-1` → balanced split closest to √dim (exact divisor pair minimizing `m+n`).
- `factor=k>0` and `dim % k == 0` → `(min(k, dim/k), max(...))`. **The small side always goes to W1** — so `factor` above √dim silently flips roles (e.g. `factorization(1024, 64) = (16, 64)`, W1 gets 16, not 64).
- `factor=k` not dividing `dim` → largest divisor ≤ k (search loop, general.py 46–56). E.g. `factorization(3072, 10) = (8, 384)`.
- `unbalanced_factorization=True` swaps only the out-dim pair (`out_l, out_k = out_k, out_l`, lines 98–99 / 146–147), making W1 `(b_out × c)` and W2 `(a_out × d)`.
- `decompose_both=True`: W1 is also low-ranked (`lokr_w1_a (a×r) @ lokr_w1_b (r×c)`) if `r < max(a, c)/2` and not `full_matrix`.
- W2 is low-ranked (`lokr_w2_a (b×r) @ lokr_w2_b (r×d)`) iff `lora_dim < max(b, d)/2 and not full_matrix`; otherwise `lokr_w2` is the full `(b×d[,k...])` matrix and, if the user did not ask for `full_matrix`, a one-time warning `logging_force_full_matrix` (lines 15–21, `@cache`d per (dim, max(in,out), factor)) fires. "dim=10000" is simply a way to trip this branch; `full_matrix=True` does it explicitly.
- `use_tucker=True` on a conv with k>1 and low-rank W2: `lokr_t2 (r×r×k1×k2)`, `lokr_w2_a (r×b)`, `lokr_w2_b (r×d)`, rebuilt by `rebuild_tucker` einsum `"i j ..., i p, j r -> p r ..."` → `(b, d, k1, k2)`. **Note the orientation of `lokr_w2_a` flips: `(r, b)` in Tucker mode vs `(b, r)` otherwise** (line 124 vs 132).
- Non-Tucker conv low-rank: `lokr_w2_a (b×r)`, `lokr_w2_b (r × d·k1·k2)` flattened (line 132–137).

**Verbatim — `LokrModule.__init__` parameter creation (lokr.py 82–174):**
```python
82	        factor = int(factor)
83	        self.lora_dim = lora_dim
84	        self.tucker = False
85	        self.use_w1 = False
86	        self.use_w2 = False
87	        self.full_matrix = full_matrix
88	        self.rs_lora = rs_lora
89	
90	        if self.module_type.startswith("conv"):
91	            in_dim = org_module.in_channels // org_module.groups
92	            k_size = org_module.kernel_size
93	            out_dim = org_module.out_channels
94	            self.shape = (out_dim, in_dim, *k_size)
95	
96	            in_m, in_n = factorization(in_dim, factor)
97	            out_l, out_k = factorization(out_dim, factor)
98	            if unbalanced_factorization:
99	                out_l, out_k = out_k, out_l
100	            shape = ((out_l, out_k), (in_m, in_n), *k_size)  # ((a, b), (c, d), *k_size)
101	            self.tucker = use_tucker and any(i != 1 for i in k_size)
102	            if (
103	                decompose_both
104	                and lora_dim < max(shape[0][0], shape[1][0]) / 2
105	                and not self.full_matrix
106	            ):
107	                self.lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], lora_dim))
108	                self.lokr_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1][0]))
109	            else:
110	                self.use_w1 = True
111	                self.lokr_w1 = nn.Parameter(
112	                    torch.empty(shape[0][0], shape[1][0])
113	                )  # a*c, 1-mode
114	
115	            if lora_dim >= max(shape[0][1], shape[1][1]) / 2 or self.full_matrix:
116	                if not self.full_matrix:
117	                    logging_force_full_matrix(lora_dim, max(in_dim, out_dim), factor)
118	                self.use_w2 = True
119	                self.lokr_w2 = nn.Parameter(
120	                    torch.empty(shape[0][1], shape[1][1], *k_size)
121	                )
122	            elif self.tucker:
123	                self.lokr_t2 = nn.Parameter(torch.empty(lora_dim, lora_dim, *shape[2:]))
124	                self.lokr_w2_a = nn.Parameter(
125	                    torch.empty(lora_dim, shape[0][1])
126	                )  # b, 1-mode
127	                self.lokr_w2_b = nn.Parameter(
128	                    torch.empty(lora_dim, shape[1][1])
129	                )  # d, 2-mode
130	            else:  # Conv2d not tucker
131	                # bigger part. weight and LoRA. [b, dim] x [dim, d*k1*k2]
132	                self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], lora_dim))
133	                self.lokr_w2_b = nn.Parameter(
134	                    torch.empty(
135	                        lora_dim, shape[1][1] * torch.tensor(shape[2:]).prod().item()
136	                    )
137	                )
138	                # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d*k1*k2)) = (a, b)⊗(c, d*k1*k2) = (ac, bd*k1*k2)
139	        else:  # Linear
140	            in_dim = org_module.in_features
141	            out_dim = org_module.out_features
142	            self.shape = (out_dim, in_dim)
143	
144	            in_m, in_n = factorization(in_dim, factor)
145	            out_l, out_k = factorization(out_dim, factor)
146	            if unbalanced_factorization:
147	                out_l, out_k = out_k, out_l
148	            shape = (
149	                (out_l, out_k),
150	                (in_m, in_n),
151	            )  # ((a, b), (c, d)), out_dim = a*c, in_dim = b*d
152	            # smaller part. weight scale
153	            if (
154	                decompose_both
155	                and lora_dim < max(shape[0][0], shape[1][0]) / 2
156	                and not self.full_matrix
157	            ):
158	                self.lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], lora_dim))
159	                self.lokr_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1][0]))
160	            else:
161	                self.use_w1 = True
162	                self.lokr_w1 = nn.Parameter(
163	                    torch.empty(shape[0][0], shape[1][0])
164	                )  # a*c, 1-mode
165	            if lora_dim < max(shape[0][1], shape[1][1]) / 2 and not self.full_matrix:
166	                # bigger part. weight and LoRA. [b, dim] x [dim, d]
167	                self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], lora_dim))
168	                self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, shape[1][1]))
169	                # w1 ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d)) = (a, b)⊗(c, d) = (ac, bd)
170	            else:
171	                if not self.full_matrix:
172	                    logging_force_full_matrix(lora_dim, max(in_dim, out_dim), factor)
173	                self.use_w2 = True
174	                self.lokr_w2 = nn.Parameter(torch.empty(shape[0][1], shape[1][1]))
```

### 2.3 alpha / scale / `rs_lora` (lines 207–220) and `use_scalar` (222–225)
```python
207	        if isinstance(alpha, torch.Tensor):
208	            alpha = alpha.detach().float().numpy()
209	        alpha = lora_dim if alpha is None or alpha == 0 else alpha
210	        if self.use_w2 and self.use_w1:
211	            # use scale = 1
212	            alpha = lora_dim
213	
214	        r_factor = lora_dim
215	        if self.rs_lora:
216	            r_factor = math.sqrt(r_factor)
217	
218	        self.scale = alpha / r_factor
219	
220	        self.register_buffer("alpha", torch.tensor(alpha * (lora_dim / r_factor)))
```
- `scale = alpha / dim` (or `alpha / √dim` with `rs_lora`). When **both** W1 and W2 are full matrices (`full_matrix` or forced), `alpha := lora_dim` so `scale = 1` and user alpha is ignored (documented in docs/algorithms/README.md line 36).
- The saved `alpha` buffer is `alpha · dim / r_factor`, i.e. always "effective scale × dim". With `rs_lora`, the file's alpha is `alpha·√dim`, so a third-party loader computing `alpha/dim` reproduces `alpha/√dim`. For full-matrix LoKr the buffer equals `lora_dim` (e.g. `10000`).
- `use_scalar=True`: `self.scalar = nn.Parameter(0.0)` and all factors get random init; else `scalar` is a non-persistent buffer `1.0`. `custom_state_dict` folds `scalar` into `lokr_w1` (or `lokr_w1_a`) on save (lines 405–409) and `load_weight_hook` (345–357) resets scalar to 1 and strips `scalar` from `missing_keys`.

### 2.4 Initialization (lines 227–245)
| Param | `use_scalar=False` | `use_scalar=True` |
|---|---|---|
| `lokr_w1` / `lokr_w1_a`, `lokr_w1_b` | kaiming_uniform(a=√5) | same |
| `lokr_w2` (full) | **zeros** | kaiming_uniform |
| `lokr_w2_a` | kaiming_uniform | kaiming_uniform |
| `lokr_w2_b` | **zeros** | kaiming_uniform |
| `lokr_t2` | kaiming_uniform | kaiming_uniform |
| `scalar` | buffer 1.0 | Parameter **0.0** |

So ΔW = 0 at step 0 either because the W2 side is zero or because `scalar = 0`. `kaiming_uniform_(a=√5)` gives bound `1/√fan_in` with `fan_in = shape[1]` (identical to `nn.Linear` default init) — for `lokr_w2_a (b×r)` that is `1/√r`, for `lokr_w1 (a×c)` it is `1/√c`.

### 2.5 Forward paths
**Non-bypass ("rebuild") — `forward` → `_rebuild_forward` (578–586, 560–576):**
```python
560	    def _rebuild_forward(self, x, *args, **kwargs):
561	        base = self.org_forward(x, *args, **kwargs)
562	        base_weight = self._current_weight().to(x.device)
563	        diff_weight = self.get_weight(self.shape).to(base_weight.dtype) * self.scalar
564	
565	        if self.wd:
566	            new_weight = self.apply_weight_decompose(
567	                base_weight + diff_weight, self.multiplier
568	            )
569	        elif self.multiplier == 1:
570	            new_weight = base_weight + diff_weight
571	        else:
572	            new_weight = base_weight + diff_weight * self.multiplier
573	
574	        delta_weight = (new_weight - base_weight).to(device=x.device, dtype=x.dtype)
575	        delta = self.op(x, delta_weight, None, **self.kw_dict)
576	        return base + delta
```
It runs the original layer **and** a second full GEMM/conv with the materialized `ΔW` (`torch.kron` → `out×in`). This "incremental delta" formulation (commit `ac2616f`) exists so several wrappers can stack. `module_dropout` (579–581) skips the whole adapter with probability p in training.

**`get_weight` / `get_diff_weight` / `get_merged_weight` (359–395), verbatim:**
```python
359	    def get_weight(self, shape):
360	        weight = kron_weight(
361	            self.lokr_w1 if self.use_w1 else None,
362	            None if self.use_w1 else self.lokr_w1_a,
363	            None if self.use_w1 else self.lokr_w1_b,
364	            self.lokr_w2 if self.use_w2 else None,
365	            None if self.use_w2 else self.lokr_w2_a,
366	            None if self.use_w2 else self.lokr_w2_b,
367	            self.lokr_t2 if (self.tucker and not self.use_w2) else None,
368	            scale=self.scale,
369	        )
370	        dtype = weight.dtype
371	        if shape is not None:
372	            weight = weight.view(shape)
373	        if self.training and self.rank_dropout:
374	            drop = (torch.rand(weight.size(0)) > self.rank_dropout).to(dtype)
375	            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
376	            if self.rank_dropout_scale:
377	                drop /= drop.mean()
378	            weight *= drop
379	        return weight
380	
381	    def get_diff_weight(self, multiplier=1, shape=None, device=None):
382	        scale = self.scale * multiplier
383	        diff = self.get_weight(shape) * scale
384	        if device is not None:
385	            diff = diff.to(device)
386	        return diff, None
387	
388	    def get_merged_weight(self, multiplier=1, shape=None, device=None):
389	        diff = self.get_diff_weight(multiplier=1, shape=shape, device=device)[0]
390	        weight = self.org_weight
391	        if self.wd:
392	            merged = self.apply_weight_decompose(weight + diff, multiplier)
393	        else:
394	            merged = weight + diff * multiplier
395	        return merged, None
```
Observations (see §10 for consequences): `get_weight` already multiplies by `self.scale` inside `kron_weight`→`make_kron`; `get_diff_weight` multiplies by `self.scale` **again** and does **not** apply `scalar`. The forward path (`_rebuild_forward`, bypass) applies `scale` once and `scalar`. `rank_dropout` here is a row mask on ΔW (out-dim rows), generated on **CPU** with no device move (line 374).

**Bypass — `bypass_forward` (557–558) = `org_forward(x) + bypass_forward_diff(x, multiplier)`; `bypass_forward_diff` (446–555), verbatim:**
```python
446	    def bypass_forward_diff(self, h, scale=1):
447	        device = h.device
448	        dtype = h.dtype
449	
450	        def to_input_dtype(tensor):
451	            return tensor.to(device=device, dtype=dtype)
452	
453	        is_conv = self.module_type.startswith("conv")
454	        if not is_conv:
455	            # Linear: the whole grouped chain is one dispatched call, and the
456	            # fused kernel never builds kron(w1, w2).
457	            diff = kron_bypass(
458	                h,
459	                to_input_dtype(self.lokr_w1) if self.use_w1 else None,
460	                None if self.use_w1 else to_input_dtype(self.lokr_w1_a),
461	                None if self.use_w1 else to_input_dtype(self.lokr_w1_b),
462	                to_input_dtype(self.lokr_w2) if self.use_w2 else None,
463	                None if self.use_w2 else to_input_dtype(self.lokr_w2_a),
464	                None if self.use_w2 else to_input_dtype(self.lokr_w2_b),
465	                None,
466	                scale=self.scale * scale,
467	            )
468	            return self.drop(diff * self.scalar.to(device=device, dtype=dtype))
469	        if self.use_w2:
470	            ba = to_input_dtype(self.lokr_w2)
471	        else:
472	            a = self.lokr_w2_b
473	            b = self.lokr_w2_a
474	
475	            if self.tucker:
476	                t = self.lokr_t2
477	                # rebuild_tucker contracts w2a over the *output* index of t, so
478	                # w2a is stored (dim, vp) while the 1x1 conv that closes the
479	                # chain needs (vp, dim); the plain path already stores it that
480	                # way, hence the transpose here and not there.
481	                a = a.view(*a.shape, *[1] * (len(t.shape) - 2))
482	                b = b.transpose(0, 1).reshape(
483	                    b.shape[1], b.shape[0], *[1] * (len(t.shape) - 2)
484	                )
485	                t = to_input_dtype(t)
486	            elif is_conv:
487	                # w2_b is stored flattened as (dim, vq * k1 * k2 * ...), so the
488	                # kernel window comes back out of that axis rather than being
489	                # appended to it (#221).
490	                a = a.view(a.shape[0], -1, *self.shape[2:])
491	                b = b.view(*b.shape, *[1] * (len(self.shape) - 2))
492	            a = to_input_dtype(a)
493	            b = to_input_dtype(b)
494	
495	        if self.use_w1:
496	            c = self.lokr_w1
497	        else:
498	            c = self.lokr_w1_a @ self.lokr_w1_b
499	        c = to_input_dtype(c)
500	        uq = c.size(1)
501	
502	        if is_conv:
503	            # (n, uq), vq, ...
504	            # Named `n`, not `b`: `b` is the second Kronecker factor above and
505	            # unpacking the batch over it fed the batch size to conv as a
506	            # weight (#221).
507	            n, _, *rest = h.shape
508	            h_in_group = h.reshape(n * uq, -1, *rest)
509	        else:
510	            # b, ..., uq, vq
511	            h_in_group = h.reshape(*h.shape[:-1], uq, -1)
512	
513	        if self.use_w2:
514	            hb = self.op(h_in_group, ba, **self.kw_dict)
515	        else:
516	            if is_conv:
517	                if self.tucker:
518	                    ha = self.op(h_in_group, a)
519	                    ht = self.op(ha, t, **self.kw_dict)
520	                    hb = self.op(ht, b)
521	                else:
522	                    ha = self.op(h_in_group, a, **self.kw_dict)
523	                    hb = self.op(ha, b)
524	            else:
525	                ha = self.op(h_in_group, a, **self.kw_dict)
526	                hb = self.op(ha, b)
527	
528	        if is_conv:
529	            # (n, uq), vp, ..., f
530	            # -> n, uq, vp, ..., f
531	            # -> n, f, vp, ..., uq
532	            hb = hb.view(n, -1, *hb.shape[1:])
533	            h_cross_group = hb.transpose(1, -1)
534	        else:
535	            # b, ..., uq, vq
536	            # -> b, ..., vq, uq
537	            h_cross_group = hb.transpose(-1, -2)
538	
539	        hc = F.linear(h_cross_group, c)
540	        if is_conv:
541	            # n, f, vp, ..., up
542	            # -> n, up, vp, ... ,f
543	            # -> n, c, ..., f
544	            hc = hc.transpose(1, -1)
545	            h = hc.reshape(n, -1, *hc.shape[3:])
546	        else:
547	            # b, ..., vp, up
548	            # -> b, ..., up, vp
549	            # -> b, ..., c
550	            hc = hc.transpose(-1, -2)
551	            h = hc.reshape(*hc.shape[:-2], -1)
552	
553	        return self.drop(
554	            h * self.scale * scale * self.scalar.to(device=device, dtype=dtype)
555	        )
```
**Yes, the Kronecker structure is exploited — ΔW is never materialized in bypass mode.** Linear goes through `functional.lokr.kron_bypass` (eager body `_kron_bypass`, functional/lokr.py 238–318, identical algorithm to the conv code above): reshape `x (…, in)` → `(…, uq=c, vq=d)`; apply W2 (or `w2b` then `w2a`) as `F.linear` on the last axis → `(…, c, b)`; transpose → `(…, b, c)`; apply W1 (`a×c`) → `(…, b, a)`; transpose → `(…, a, b)`; flatten → `(…, a·b=out)`. That is `vec(W1 · X · W2ᵀ)`. FLOPs per token drop from `out·in` to `c·b·(d + a)` (for 3072², factor −1: 344k vs 9.4M). On CUDA with Triton/TileLang installed and all factors ≤128 it goes to a fused kernel (`kernels/triton/lokr/bypass.py`, two `tl.dot`s per token tile in registers, fp32 atomics for factor grads). `rank_dropout` is **not** applied in bypass mode; `dropout` (`self.drop`) is applied to ΔW·x only in bypass mode (the constructor's `[WARN] ... haven't implemented normal dropout` at line 201–202 is stale).

`forward` (578–586): `module_dropout` → maybe skip; if `bypass_mode and not (wd and FP8 base)` → bypass, else `_rebuild_forward` (DoRA on FP8 needs the dequantized full weight, so it forces rebuild even though FP8 forces bypass elsewhere).

### 2.6 DoRA (`weight_decompose=True`, `wd_on_out=True`) — the "DoRA apply code"
It is **not in base.py**; each of locon/loha/lokr creates `dora_scale` in its own `__init__` and calls the shared `functional.general.weight_decompose`.

Init (lokr.py 176–198):
```python
176	        self.wd = weight_decompose
177	        self.wd_on_out = wd_on_out
178	        if self.wd:
179	            org_weight = self._current_weight().cpu().clone().float()
180	            self.dora_norm_dims = org_weight.dim() - 1
181	            if self.wd_on_out:
182	                self.dora_scale = nn.Parameter(
183	                    torch.norm(
184	                        org_weight.reshape(org_weight.shape[0], -1),
185	                        dim=1,
186	                        keepdim=True,
187	                    ).reshape(org_weight.shape[0], *[1] * self.dora_norm_dims)
188	                ).float()
189	            else:
190	                self.dora_scale = nn.Parameter(
191	                    torch.norm(
192	                        org_weight.transpose(1, 0).reshape(org_weight.shape[1], -1),
193	                        dim=1,
194	                        keepdim=True,
195	                    )
196	                    .reshape(org_weight.shape[1], *[1] * self.dora_norm_dims)
197	                    .transpose(1, 0)
198	                ).float()
```
`dora_scale` = per-output-row L2 norm of the *base* weight, shape `(out, 1[,1,1])` (or per-input-column `(1, in[,1,1])` when `wd_on_out=False`, the pre-3.2 behaviour). Apply (functional/general.py 117–139, eager body; 142–162 is the dispatcher that may route to the fused `apply_dora` kernel):
```python
117	def _weight_decompose(weight, dora_scale, multiplier, wd_on_out):
118	    """W' = W · (mult·(m/‖W‖ − 1) + 1), the norm per out row or per in column."""
119	    weight = weight.to(dora_scale.dtype)
120	    norm_dims = weight.dim() - 1
121	    if wd_on_out:
122	        weight_norm = (
123	            weight.reshape(weight.shape[0], -1)
124	            .norm(dim=1)
125	            .reshape(weight.shape[0], *[1] * norm_dims)
126	        ) + torch.finfo(weight.dtype).eps
127	    else:
128	        weight_norm = (
129	            weight.transpose(0, 1)
130	            .reshape(weight.shape[1], -1)
131	            .norm(dim=1, keepdim=True)
132	            .reshape(weight.shape[1], *[1] * norm_dims)
133	            .transpose(0, 1)
134	        ) + torch.finfo(weight.dtype).eps
135	
136	    scale = dora_scale.to(weight.device) / weight_norm
137	    if multiplier != 1:
138	        scale = multiplier * (scale - 1) + 1
139	    return weight * scale
```
`W' = (W+ΔW) · m / ‖W+ΔW‖_row`, with `m = dora_scale` trainable. The `multiplier` only interpolates the *rescale* term; ΔW itself is always added at full strength when `wd=True` (lokr.py 389–392, 565–568) — so `multiplier=0` does **not** give the base model for DoRA modules. The dispatcher refuses the fused path for `wd_on_out=False` on convs. `apply_dora_scale` (general.py 165–178) is a legacy, input-dim variant exported but unused by modules.

### 2.7 Save/load
`custom_state_dict` (400–418) writes: `alpha` (always), `dora_scale` (if wd), `lokr_w1` **× scalar** or `lokr_w1_a × scalar` + `lokr_w1_b`, `lokr_w2` or `lokr_w2_a` + `lokr_w2_b` (+ `lokr_t2` if tucker). `weight_list` (32–43) = `[lokr_w1, lokr_w1_a, lokr_w1_b, lokr_w2, lokr_w2_a, lokr_w2_b, lokr_t1, lokr_t2, alpha, dora_scale]` (`lokr_t1` is legacy, never produced); `weight_list_det = [lokr_w1, lokr_w1_a]`. `make_module_from_state_dict` (247–343) infers `lora_dim` from `w1a.size(1)` or `w2a.size(1)`, infers `factor` by checking whether W1's shape equals the `factor=-1` split (287–291) and otherwise by a divisibility heuristic (293–315), then constructs and copies. `apply_max_norm` (420–444) scales each factor tensor by `ratio^(1/n_factors)` (used by kohya's `scale_weight_norms`).

### 2.8 Worked numeric examples (computed with the actual `factorization()` and the `__init__` logic above)
```
Linear(3072→3072), dim=10000, factor=-1  (also identical with full_matrix=True)
  factorization(3072,-1) = (48, 64) for both out and in  →  a=48,b=64,c=48,d=64
  lokr_w1 (48, 48) = 2,304 params   lokr_w2 (64, 64) = 4,096 params
  total 6,400 (0.068 % of 9.44 M);  use_w1=use_w2=True → alpha:=10000, scale=1.0; saved alpha=10000
  (w2 low-rank test: 10000 < max(64,64)/2=32 → False → full W2)

Linear(3072→3072), dim=16, factor=8
  factorization(3072,8) = (8, 384) → a=8,b=384,c=8,d=384
  lokr_w1 (8, 8) = 64      lokr_w2_a (384, 16) = 6,144     lokr_w2_b (16, 384) = 6,144
  total 12,352 (0.131 %);  scale = alpha/16 (alpha=1 → 0.0625; rs_lora → alpha/4); saved alpha = alpha (or 4·alpha with rs_lora)
  ΔW = scale · lokr_w1 ⊗ (lokr_w2_a @ lokr_w2_b); rank(ΔW) ≤ 8·16 = 128

Linear(3072→12288), dim=10000, factor=-1
  factorization(12288,-1) = (96, 128); factorization(3072,-1) = (48, 64)
  lokr_w1 (96, 48) = 4,608   lokr_w2 (128, 64) = 8,192   → total 12,800 (0.034 % of 37.7 M), scale=1
  (12288→3072 direction: lokr_w1 (48, 96), lokr_w2 (64, 128), same count)

Other useful points:
  3072→3072, dim=16, factor=-1 : w1 (48,48) + w2_a (64,16) + w2_b (16,64) = 4,352 (dim≥32 would force full W2)
  3072→3072, dim=16, factor=-1, decompose_both : w1_a (48,16), w1_b (16,48), w2_a (64,16), w2_b (16,64) = 3,584
  3072→3072, full, factor=4  : w1 (4,4) + w2 (768,768) = 589,840 (6.25 %)
  3072→3072, full, factor=16 : w1 (16,16) + w2 (192,192) = 37,120 (0.39 %)
  3072→12288, dim=16, factor=8 : w1 (8,8) + w2_a (1536,16) + w2_b (16,384) = 30,784
  3072→3072, full, factor=8, unbalanced : w1 (384,8) + w2 (8,384) = 6,144
  1024→1024 full factor=-1 : (32,32)+(32,32) = 2,048 ;  1536→1536 : (32,32)+(48,48) = 3,328
```
Fused-kernel note: the Triton/TileLang bypass kernel only accepts factors ≤ 128 (`MAX_KRON_FACTOR`), so the `factor=-1` cases above are eligible, while the popular "factor=4–8 full-matrix" configuration (W2 = 384–768 wide) is **not** and falls back to torch.compile/eager (`_apply_supported`, functional/lokr.py 174–189).

---

## 3. `lycoris/functional/`

### 3.1 `general.py`
- `FUNC_LIST = [None, None, F.linear, F.conv1d, F.conv2d, F.conv3d]` (line 9) — indexed by weight `.dim()`.
- `rebuild_tucker(t, wa, wb)` (12–14): `einsum("i j ..., i p, j r -> p r ...")`.
- `factorization(dimension, factor=-1)` (17–59), verbatim:
```python
17	def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
...
38	    if factor > 0 and (dimension % factor) == 0:
39	        m = factor
40	        n = dimension // factor
41	        if m > n:
42	            n, m = m, n
43	        return m, n
44	    if factor < 0:
45	        factor = dimension
46	    m, n = 1, dimension
47	    length = m + n
48	    while m < n:
49	        new_m = m + 1
50	        while dimension % new_m != 0:
51	            new_m += 1
52	        new_n = dimension // new_m
53	        if new_m + new_n > length or new_m > factor:
54	            break
55	        else:
56	            m, n = new_m, new_n
57	    if m > n:
58	        n, m = m, n
59	    return m, n
```
  Table (rows = dim, columns = factor; produced by running the function):
```
   dim |    f= -1    |    f=  2    |    f=  4    |    f=  8    |    f= 16    |    f= 32    |    f= 64
  1024 |    (32, 32) |    (2, 512) |    (4, 256) |    (8, 128) |    (16, 64) |    (32, 32) |    (16, 64)
  1536 |    (32, 48) |    (2, 768) |    (4, 384) |    (8, 192) |    (16, 96) |    (32, 48) |    (24, 64)
  3072 |    (48, 64) |   (2, 1536) |    (4, 768) |    (8, 384) |   (16, 192) |    (32, 96) |    (48, 64)
  4096 |    (64, 64) |   (2, 2048) |   (4, 1024) |    (8, 512) |   (16, 256) |   (32, 128) |    (64, 64)
 12288 |   (96, 128) |   (2, 6144) |   (4, 3072) |   (8, 1536) |   (16, 768) |   (32, 384) |   (64, 192)
  2048 |    (32, 64) |   (2, 1024) |    (4, 512) |    (8, 256) |   (16, 128) |    (32, 64) |    (32, 64)
   768 |    (24, 32) |    (2, 384) |    (4, 192) |     (8, 96) |    (16, 48) |    (24, 32) |    (12, 64)
   320 |    (16, 20) |    (2, 160) |     (4, 80) |     (8, 40) |    (16, 20) |    (10, 32) |     (5, 64)
   127 |    (1, 127) |    (1, 127) |    (1, 127) |    (1, 127) |    (1, 127) |    (1, 127) |    (1, 127)
   360 |    (18, 20) |    (2, 180) |     (4, 90) |     (8, 45) |    (15, 24) |    (18, 20) |    (18, 20)
```
  Note the docstring example `360 -> 12, 30` for factor 16 is wrong (actual `(15, 24)`); primes degrade to `(1, dim)` (W1 becomes 1×1, W2 the whole matrix → LoKr degenerates to LoRA-with-a-scalar).
- `power2factorization` (62–84): `m` even, `n` power of two — used by BOFT.
- `tucker_weight_from_conv`, `tucker_weight` (87–95).
- `add_scaled(base, delta, gamma)` (98–114): fused `W + γΔW` for full/norm.
- `weight_decompose` (117–162): DoRA epilogue (see §2.6).

### 3.2 `lokr.py`
- `make_kron(w1, w2, scale)` (16–25): unsqueeze `w1` to `w2.dim()`, `torch.kron`, multiply by `scale` if ≠ 1.
- `weight_gen(org_weight, rank, tucker=True, factor=-1, decompose_both=False, full_matrix=False, unbalanced_factorization=False)` (28–126): functional twin of `__init__`; returns `(w1, w1a, w1b, w2, w2a, w2b, t2)` with the same init. Differences from the module: for **Linear it ignores `full_matrix`** (lines 97, 103 have no `full_matrix` test), conv `w2b` is created unflattened `(rank, d, *k)`.
- `_kron_weight` (129–140): rebuilds each half (`w1a@w1b`, `w2a@w2b.view(r,-1)` or tucker) then `make_kron`.
- `kron_weight(..., scale, backend)` (192–210): dispatch: fused `lokr_kron_weight` if plain 2-D W2 & static scale & CUDA; else torch.compile'd or eager body.
- `diff_weight(*weights, gamma=1.0)` (213–235): `scale = rank_scale(w1a, w2a, gamma) = gamma / rank` where `rank` is read from the factorized side (`kernels/autograd/lokr.py` 85–97) — so the functional API's `gamma` is *alpha*, not scale; for both-full it is `gamma/gamma = 1`.
- `_kron_bypass` (238–318) and `kron_bypass` (321–346): described in §2.5. `_apply_factor_cap` / `_apply_supported` (152–189) read `MAX_KRON_FACTOR` from the available kernel modules (guarded by `torch.compiler.is_compiling()` since `b42d907`).
- `bypass_forward_diff(h, org_out, *weights, gamma, extra_args)` (349–378): `scale = gamma / rank` (rank from `w1b`/`w2b` rows).

### 3.3 Others
- `locon.py`: `weight_gen` (down kaiming, up zero, optional tucker mid), `_diff_weight` (`gamma` multiplied into `up`), `_bypass_diff`, dispatchers.
- `loha.py`: custom `torch.autograd.Function`s `HadaWeight`/`HadaWeightTucker` (12–79) that recompute `B` and `A` in backward to avoid caching both (docs/algorithms/details.md 94–99); `weight_gen` init `w1d ~ N(0,1)`, `w1u = 0`, `w2d ~ N(0,1)`, `w2u ~ N(0,0.1)`.
- `diag_oft.py`, `boft.py`: OFT rebuild/bypass functional forms.
- `kernels/autograd/lokr.py`: `KronGenRebuildFn` (fused merge; generates `w1=w1a@w1b`, `w2=w2a@w2b` inside the kernel) and `KronApplyFn` (fused bypass); `lokr_kron_weight` / `lokr_kron_bypass` public wrappers; conv-spatial and Tucker W2 route to torch.

---

## 4. `lycoris/wrapper.py` and `lycoris/kohya.py`

### 4.1 `create_lycoris(module, multiplier=1.0, linear_dim=4, linear_alpha=1, warn_on_unmatched=True, **kwargs)` (wrapper.py 53–144)
Parses kwargs: `conv_dim`, `conv_alpha`, `dropout`, `rank_dropout`, `module_dropout`, `algo` (default `"lora"`), `use_tucker` (+ deprecated `disable_conv_cp`, `use_conv_cp`, `use_cp`), `use_scalar`, `block_size`, `train_norm`, `constraint` (+deprecated `constrain`), `rescaled`, **`dora_wd`** → `weight_decompose`, **`wd_on_output`** → `wd_on_out`, `full_matrix`, `bypass_mode` (default False), `unbalanced_factorization`, `train_llm_adapter`, `preset` (name in `PRESET` or a `.toml` path via `read_preset`), and passes `decompose_both` and `factor` **raw** (`kwargs.get(...)`). Booleans go through `str_bool` (`utils/__init__.py` 44–45: `str(val).lower() != "false"` — so `"0"`, `"no"`, `0`... are all True). **`rs_lora` is not read here** (only in kohya). It calls `LycorisNetwork.apply_preset(preset)` — mutating class state — then constructs `LycorisNetwork`.

### 4.2 `LycorisNetwork` class-level preset state (196–237)
Defaults: `ENABLE_CONV=True`, `TARGET_REPLACE_MODULE=["Linear","Conv1d","Conv2d","Conv3d","GroupNorm","LayerNorm"]`, `TARGET_REPLACE_NAME=[]`, `LORA_PREFIX="lycoris"`, `MODULE_ALGO_MAP={}`, `NAME_ALGO_MAP={}`, `USE_FNMATCH=False`, `TARGET_EXCLUDE_NAME=[]`. `apply_preset` (213–237) validates keys against `VALID_PRESET_KEYS` (config_sdk.py 7–20) and assigns `enable_conv`, `target_module`, `target_name`, `module_algo_map`, `name_algo_map`, `lora_prefix`, `use_fnmatch`, `exclude_name`. The `unet_*`/`text_encoder_*` keys are accepted but ignored by this class (they are for `LycorisNetworkKohya`). Keys not present in the preset are **left as they were** — state leaks across calls; tests reset explicitly (`test/wrapper.py` 15–34).

### 4.3 Module-tree walk (`__init__`, 239–563)
- Targets = `TARGET_REPLACE_MODULE ∪ MODULE_ALGO_MAP.keys()`; names = `TARGET_REPLACE_NAME ∪ NAME_ALGO_MAP.keys()` (500–515). `"LLMAdapterTransformerBlock"` is removed unless `train_llm_adapter` (517–519).
- `create_modules` (427–498): for each `(name, module)` in `root.named_modules()`: skip if `is_excluded(name)` (regex/fnmatch via `match_fn`, 565–575); if the class name is a target class **and** the name does not match a name pattern → recurse with `create_modules_` (365–424) using `MODULE_ALGO_MAP[class]` as the config (`algo` and any extra keys such as `dim`, `alpha`, `factor`, `use_tucker`…); elif the name matches a target name/pattern → `find_conf_for_name` (577–588) from `NAME_ALGO_MAP` (exact key first, then pattern match) or `MODULE_ALGO_MAP`, and create a single module.
- `create_modules_` recurses into children; a child whose class is in `MODULE_ALGO_MAP` (and is not the root) switches config/algo for that subtree (385–403). Names: `lora_name = (prefix + "." + name).replace(".", "_")`, prefix = `f"{LORA_PREFIX}_{top-level name}"`, e.g. `lycoris_transformer_blocks_0_attn_to_q`.
- `create_single_module` (303–363): fills missing per-module kwargs from `root_kwargs` (network-level: `decompose_both, factor, block_size, constraint, rescaled, weight_decompose, wd_on_out, full_matrix, bypass_mode, unbalanced_factorization, use_scalar, ...`); `train_norm and "Norm" in class name` → `NormModule`; supported Linear (`is_supported_linear_module`, base.py 34–41: FP8 only for lora/locon/loha/lokr/glora and DoRA only for lokr) with network `lora_dim > 0` → `dim = dim or lora_dim`, `alpha = alpha or self.alpha`; Conv with `k==1` uses linear dim/alpha, else `conv_lora_dim`/`conv_alpha` (0 disables); then `network_module_dict[algo](lora_name, module, multiplier, dim, alpha, dropout, rank_dropout, module_dropout, use_tucker, **kwargs)`.
- `network_module_dict` (32–44): lora/locon→`LoConModule`, loha, lokr, dylora, glora, full, diag-oft, boft, ia3, tlora.
- Post-checks: unmatched-target warnings (531–547), algo table log, duplicate-name assertion (559–563).

### 4.4 Runtime API
`set_multiplier` (590–593), `load_weights(file)` (595–608, safetensors or torch, `strict=False`), `apply_to()` (610–621: each `lora.apply_to()` + `self.add_module(lora_name, lora)` so `network.state_dict()` keys are `f"{lora_name}.{key}"`; re-loads `weights_sd` if `load_weights` was called first), `is_mergeable` (FP8 check), `restore`, `merge_to(weight, precise)`, `onfly_merge/restore`, `apply_max_norm_regularization` (652–665), `enable_gradient_checkpointing` (no-op marker), `prepare_optimizer_params(lr)` (676–690: **one** param group, all adapter params, `requires_grad_(True)`), `save_weights(file, dtype, metadata)` (701–721: casts every tensor to `dtype` on CPU; safetensors with metadata dict — **no hash added** in the generic wrapper).

### 4.5 `create_lycoris_from_weights(multiplier, file, module, weights_sd=None)` (147–193)
Groups keys by `key.split(".")[0]`, maps `f"{LORA_PREFIX}_{name}".replace(".","_")` over `module.named_modules()`, then `get_module` (detect algo by key names) + `make_module` (`make_module_from_state_dict`) per module. Modules are created with the **current** `LORA_PREFIX`.

### 4.6 `lycoris/kohya.py`
- `create_network(multiplier, network_dim, network_alpha, vae, text_encoder, unet, warn_on_unmatched=True, **kwargs)` (31–156) — the sd-scripts entry point (`--network_module lycoris.kohya`). Recognized `network_args` (all arrive as strings): `conv_dim, conv_alpha, dropout, rank_dropout, module_dropout, algo, use_tucker (use_cp/use_conv_cp/disable_conv_cp deprecated), use_scalar, block_size, train_norm, constraint (constrain), rescaled, dora_wd, wd_on_output, full_matrix, bypass_mode, rs_lora, unbalanced_factorization, train_t5xxl, train_llm_adapter, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio, preset, decompose_both, factor`. **Not recognized:** `weight_decompose`, `wd_on_out`, `rank_dropout_scale`, `unbalanced` short forms. `decompose_both` is passed raw (string `"False"` is truthy → enables it). `factor` is `int()`-ed in the module.
- `LycorisNetworkKohya` (248–901): separate class attributes `UNET_TARGET_REPLACE_MODULE` (255–282: `Transformer2DModel, ResnetBlock2D, Downsample2D, Upsample2D, HunYuanDiTBlock, DoubleStreamBlock, SingleStreamBlock, SingleDiTBlock, MMDouble/SingleStreamBlock, WanAttentionBlock, HunyuanVideo*, JointTransformerBlock, FinalLayer, QwenImageTransformerBlock, LensTransformerBlock, Ideogram4TransformerBlock, ZImageTransformerBlock, AceStep*, TextFusionBlock, Block, PatchEmbed, TimestepEmbedding, LLMAdapterTransformerBlock`), `UNET_TARGET_REPLACE_NAME` (`conv_in, conv_out, time_embedding.linear_1/2`), `TEXT_ENCODER_TARGET_REPLACE_MODULE` (CLIP*, MT5Block, BertLayer, Gemma2*, Qwen3*), prefixes **`LORA_PREFIX_UNET="lora_unet"`, `LORA_PREFIX_TEXT_ENCODER="lora_te"`** (`lora_te1_`/`lora_te2_` when `text_encoder` is a list, 584–591). `apply_preset` (312–334) reads the `unet_*`/`text_encoder_*` keys. `apply_to(text_encoder, unet, apply_text_encoder, apply_unet)` (718–742). `merge_to(text_encoder, unet, weights_sd, dtype, device)` (745–764). `prepare_optimizer_params(text_encoder_lr, unet_lr, learning_rate)` (795–860) returns `(param_groups, descriptions)` with LoRA+ support keyed on `"lora_up" in param name` (so only LoCon gets a "plus" group). `save_weights` (878–901) adds `sshs_model_hash` (sha256 over tensor bytes, `utils/__init__.py` 33–41) to safetensors metadata.
- `create_network_from_weights` (159–245): same as the wrapper variant but split by prefix.
- **`is_sdxl` / `is_v2`:** not in the library; they only appear in `tools/extract_locon.py` and `tools/merge.py` as CLI flags to pick sd-scripts' `load_models_from_stable_diffusion_checkpoint(is_v2, ...)` vs `load_models_from_sdxl_checkpoint`. SDXL-ness in the network itself is only "text_encoder is a list → `lora_te1_`/`lora_te2_`".

---

## 5. Other algorithms (`lycoris/modules/`)

| Module (file) | Parameters / saved keys | Math | bypass | wd (DoRA) | conv | use_scalar |
|---|---|---|---|---|---|---|
| `LoConModule` (locon.py, algo `lora`/`locon`) | `lora_down.weight (r×in[×k])`, `lora_up.weight (out×r[×1])`, `lora_mid.weight (r×r×k)` tucker, `alpha`, `dora_scale` | `ΔW = scale·up@down` (or tucker chain); rank_dropout masks the r axis in bypass | yes (grouped conv forces rebuild, 87–91) | yes | yes | yes (folded into `lora_up`) |
| `LohaModule` (loha.py) | `hada_w1_a (out×r)`, `hada_w1_b (r×in)`, `hada_w2_a`, `hada_w2_b`, `hada_t1/t2` tucker, `alpha`, `dora_scale` | `ΔW = scale·(w1a@w1b) ⊙ (w2a@w2b)`, custom backward; init `w1_b~N(0,1), w1_a~N(0,0.1), w2_b~N(0,1), w2_a=0` | yes (linear non-tucker fused; else rebuild ΔW then op) | yes | yes | yes |
| `LokrModule` (lokr.py) | see §2 | Kronecker | yes | yes (also on FP8 base) | yes | yes |
| `FullModule` (full.py, algo `full`) | `diff`, `diff_b` | `apply_to` **removes** `weight`/`bias` from the base module and trains the absolute weight; saves `weight − org` as `diff` | no (raises; overrides auto-bypass) | no | yes | no |
| `NormModule` (norms.py, via `train_norm=True`) | `w_norm`, `b_norm` (additive deltas on LayerNorm/GroupNorm affine) | `γ' = γ + w_norm` | n/a | no | n/a | no |
| `DiagOFTModule` (diag_oft.py, `diag-oft`) | `oft_blocks (block_num×bs×bs)`, `rescale (out,1..)`, `alpha` (=constraint) | `factorization(out, lora_dim) = (block_size, block_num)`; `Q = blocks − blocksᵀ`, `R = (I+Q)(I−Q)⁻¹` (Cayley), `W' = R·W` blockwise; `constraint` (COFT) clamps ‖Q‖, `rescaled` adds per-out scale | yes (applied on the output) | no | yes | no |
| `ButterflyOFTModule` (boft.py, `boft`) | `oft_blocks (m×block_num×b×b)`, `rescale`, `alpha` | `power2factorization`; butterfly product of `m = popcount(...)+1` block-orthogonal stages | yes | no | yes | no |
| `GLoRAModule` (glora.py) | `a1.weight, a2.weight, b1.weight, b2.weight, bm.weight, alpha` | `ΔW = scale·(W·a1·a2 + b1·b2)`; bypass `W(x + A(x)) + B(x)` | yes | no | yes | yes |
| `IA3Module` (ia3.py) | `weight (out or in vector)`, `on_input` buffer | `W' = W·(1 + w)` on out or in channels; `train_on_input` per-name via the `ia3` preset | yes | no | yes | no |
| `DyLoraModule` (dylora.py) | saves as `lora_up.weight`/`lora_down.weight`/`alpha` (LoRA-compatible); internally `up_list`/`down_list` of `block_size` blocks | random rank prefix each step; `scale = alpha/(b+1)`; `load_state_dict` is a no-op stub (88–89) | nominally (buggy: reshapes to full `lora_dim`, `scale` unused — 144–152) | no | yes | no |
| `TLoraModule` (tlora.py, `tlora`) | `q_layer.weight, p_layer.weight, lambda_layer (1×r), alpha` + frozen `base_q/base_p/base_lambda` buffers | SVD init of W; `ΔW = P·diag(λ⊙mask)·Q − P₀·diag(λ₀⊙mask)·Q₀`; timestep mask via global `set_timestep_mask` | yes | no | 1×1 only (k>1 falls back to bypass) | no |

`config_sdk.ALGO_REGISTRY` (config_sdk.py 37–175) lists per-algo accepted preset options (used only when `PresetConfig.from_dict(..., strict=True)`, which `read_preset` does not enable).

---

## 6. Merging / extraction / conversion

- `lycoris/utils/__init__.py`:
  - `extract_linear` (109–154) / `extract_conv` (60–106): SVD of `W_db − W_base`, rank selection modes `fixed | threshold | ratio | quantile`, `U·S` → up, `Vh` → down; returns `"full"` when `rank ≥ out/2`; conv optionally re-decomposed into a Tucker `lora_mid` (`small_conv`, 247–275).
  - `extract_diff(base_tes, db_tes, base_unet, db_unet, mode, linear_mode_param, conv_mode_param, extract_device, use_bias, sparsity, small_conv)` (157–353): emits **LoCon-format** keys under `lora_unet_`/`lora_te[N]_`: `.lora_down.weight`, `.lora_up.weight`, `.alpha` (= rank), optional sparse bias `bias_indices/bias_values/bias_size` (`make_sparse`, 52–57), or `diff`/`diff_b` for full and `w_norm`/`b_norm` for norms. **There is no LoKr/LoHa extraction** — only SVD→LoRA.
  - `merge(tes, unet, lyco_state_dict, scale, device)` (420–483): detects each module's algo via `get_module`, rebuilds it with `make_module`, and calls `module.merge_to(scale)`; converts diffusers-style kohya names to SGM names for SDXL via `convert_diffusers_name_to_compvis` (372–417, e.g. `lora_unet_down_blocks_0_attentions_0_…` → `lora_unet_input_blocks_1_1_…`).
  - `precalculate_safetensors_hashes` (33–41).
- `tools/extract_locon.py` (CLI over `extract_diff`; needs sd-scripts `library` on `sys.path`; `--is_v2`, `--is_sdxl`, `--mode`, `--linear_dim/--conv_dim/...`, `--use_sparse_bias`, `--disable_cp`), `tools/merge.py` (CLI over `merge`; `--weight`, `--dtype`, `--is_sdxl`, `--is_v2`), `tools/batch_hcp_convert.py` (HCP-Diffusion ↔ webui key format; prefixes `lora_unet_`, `lora_te_`, `lora_te1_`, `lora_te2_`), `tools/batch_bundle_convert.py` (embedding bundles), `tools/pack_bundle.py`, `tools/sdxl_emb.py`.
- **LyCORIS file format (what ComfyUI/webui consume):** one flat safetensors dict, keys `<prefix>_<module path with "." → "_">.<param>`, `alpha` a 0-d tensor per module, `dora_scale` `(out,1[,1,1])` when DoRA. Kohya path prefixes: `lora_unet_`, `lora_te_` / `lora_te1_` / `lora_te2_`; generic wrapper prefix: `lycoris_` (or whatever `lora_prefix` the preset sets). LoKr keys: `lokr_w1 | lokr_w1_a + lokr_w1_b`, `lokr_w2 | lokr_w2_a + lokr_w2_b [+ lokr_t2]`, `alpha`, `dora_scale`. The repo contains no ComfyUI documentation (only a mention in README.md line 67). From my knowledge of those loaders (verify against their source): they compute `scale = alpha / dim` with `dim` read from the rank axis of `lokr_w2_b` (or `lokr_w1_b`), and use `scale = 1` when both halves are full — which is exactly why LyCORIS forces `alpha := lora_dim` in that case and stores `alpha·dim/√dim` for `rs_lora`; the `scalar` must be folded into `lokr_w1` at save time (it is) because loaders have no `scalar` key.

---

## 7. Presets and documented hyperparameters

- Built-in presets (`lycoris/config.py` 58–212, `PresetConfig` dataclass in `config_sdk.py` 223–307): `full` (enable_conv, all UNet block classes + `conv_in/conv_out/time_embedding.*`, all TE attention/MLP classes), `full-lin` (no conv), `attn-mlp` (transformer blocks only, "kohya preset"), `attn-only` (`CrossAttention/SelfAttention` + TE attentions), `unet-only`, `unet-transformer-only`, `unet-convblock-only`, `ia3` (`to_k, to_v, ff.net.2`, `k_proj, v_proj, mlp.fc2`, with `name_algo_map` forcing `train_on_input=True` for the MLP outputs).
- TOML preset fields (`docs/usage/presets.md`, `example_configs/preset_configs/example.toml`): `enable_conv`, `unet_target_module`, `unet_target_name` (regex), `text_encoder_target_module`, `text_encoder_target_name`, `module_algo_map.<Class>` / `name_algo_map.<name or pattern>` tables with `algo` + per-module overrides (`dim`, `alpha`, `factor`, `use_tucker`, …), `use_fnmatch`, `exclude_name`, `lora_prefix`, `target_module`/`target_name` (generic wrapper). The example maps `CrossAttention → lokr dim=1e13 factor=64`, `FeedForward → lokr dim=1e11 factor=12`, `ResnetBlock2D → lora dim=4 alpha=1 use_tucker`, `CLIPAttention → lora dim=8 alpha=1`, `CLIPMLP → lokr full`.
- Recommended LoKr hyperparameters from docs:
  - `docs/algorithms/README.md` 29–39: "Small LoKr: factor=-1 (~900–2500 KB)", "Large LoKr: factor≈8, full dimension (LoRA-like)"; full dimension via a huge dim (e.g. 10000), alpha ignored then; small LoKr may not transfer across base models; smaller factor → bigger file.
  - `docs/algorithms/guidelines.md` 30–35: if a LoRA "does not learn well enough", use LoKr with **low factors 4–8 and full dimension (dim=100000)**; if it "learns too well", use LoHa or LoKr with **large factors / lower dim**; `preset=attn-mlp` recommended in most cases; SD1 size table: `LoKr attn-mlp full[-1]` 1.6 MB, `8[4]` 2.8 MB, `full[8]` 12 MB, `full[4]` 43 MB.
  - README.md 45: "low factor" ≈ `factor ≤ 0.5·√dim` (≤8 for SD-class models), "high factor" ≈ `factor ≥ √dim` (≥16).
  - `docs/usage/network-args.md` 18–30: dim > 10240/2 prevents W2 decomposition; alpha ignored by full-dim LoKr; merge ratio `alpha/dim`; `dora_wd=True` forces `bypass_mode=False`.
  - Example configs: `example_configs/training_configs/kohya/lokr_config.toml` (`network_dim=100000`, `network_alpha=1`, `algo=lokr factor=6 preset=attn-mlp`, `unet_lr=text_encoder_lr=2e-4`, AdamW8bit, bf16, batch 8); `hcp_lokr.yaml` (`dim=10000, alpha=0, factor=8, lr=1e-4`).
  - LoHa caveat (README 27): high dim → NaN, lower LR. No LoKr-specific LR caveat is documented beyond the examples (1e-4 – 2e-4).

---

## 8. Kernel dispatch and `torch.compile`

- **Order:** `triton > tilelang > compile > torch` (`kernels/dispatch.py` 17–20, `resolve_backend` 57–66). `available_backends()` probes importability once; `LYCORIS_KERNEL_BACKEND=auto|triton|tilelang|compile|torch` overrides; `demote(name)` drops a backend after a runtime failure (`select.call_compiled`, 109–115). `TORCHDYNAMO_DISABLE` disables the compile tier.
- **Per-call choice:** `select.choose(tensors, supported, backend)` (61–86): CPU → `torch`; CUDA + all-floating same-device + not fp16/bf16 mixed + caller's `supported` → fused; else `compile` (only on CUDA unless asked by name) or `torch`. `static_scale` (89–95) forbids fusing when the scale is a tensor with grad. `compiled(fn)` (98–106) = `torch.compile(fn, dynamic=None)` cached per function object — compile is applied **per op**, never to the module.
- **Recent commits:** `b42d907` "make kernel dispatch torch.compile safe": `choose()` returns `"torch"` when `torch.compiler.is_compiling()` (so an *outer* `torch.compile` of the user's model captures the eager body instead of nesting a compile or importing Triton during tracing), and `_apply_supported` short-circuits the lazy backend import while compiling; added `test/torch_compile.py`. It also added a `torch.randint`-based masked variant of DyLoRA's random rank; `d5ed22b` "preserve DyLoRA gradient routing" **reverted** that DyLoRA change (masking every block changed which parameters receive gradients) and deleted its test. `573b799` added `MAX_KRON_FACTOR=128` scope gating for the LoKr bypass kernel. `ff4a594` (#221) fixed the LoKr conv bypass chain. `82df4aa` (#228) fixed `algo=full` after `apply_to`. `2fc7c60` (#288) fixed `exclude_name` handling.
- **Custom kernels:** `lycoris/kernels/triton/{lora,loha,lokr,oft,boft,dora,ia3,merge}` and TileLang twins; 23 ops listed in `kernels/triton/ops.py`. LoKr: `lokr_merge_fwd/bwd` (output-tiled Kronecker gather, factorized sides generated in-kernel), `lokr_full_merge_*`, `lokr_bypass_fwd/bwd` (register-resident two-dot per token tile). Tile shapes come from an analytic planner + measured tuner (`kernels/plans/`, cache at `~/.cache/lycoris_kernels/tuning.json`, `LYCORIS_KERNEL_TUNE=off`). Precision policy (`kernels/precision.py`): 16-bit MMA with fp32 accumulate, output = promoted dtype, grads returned in each leaf's dtype.
- **Automatic bypass for quantized bases:** §1.2 — FP8 duck-type, bnb/quanto `QuantLinears`, or any non-`"Linear"`-named linear-like when `bypass_mode is None`. `FullModule` overrides the auto-decision back to rebuild (full.py 58–60).

---

## 9. Testing

`unit-test.py` runs (with coverage): `test/module.py` (every module × Linear/Conv1d/2d/3d × device/dtype × wd × tucker × scalar: apply_to → forward → backward → `apply_max_norm` → state_dict round-trip → restore → `merge_to`; bypass and parametrize variants; FP8 duck-type tests including `lokr` + DoRA on FP8 matching a dequantized reference), `test/wrapper.py` (`create_lycoris` → forward → restore → `merge_to` → `create_lycoris_from_weights` → forward equality; multi-wrapper stacking; regex / fnmatch / exclude presets; diffusers Flux model targeting), `test/functional.py` (`diff_weight` vs `bypass_forward_diff` consistency per functional module), `test/kohya.py` (`create_network` over sd-scripts models, `extract_diff`), `test/precision_merge_test.py` (fp64 precise merge drift), `test/kernels/test_ops.py` + `test_autograd.py` (fused vs fp64 / eager parity, CUDA only), `test/torch_compile.py`. Ad-hoc scripts: `test/restore.py`, `test/compile.py`, `test/grouped_conv_locon.py`, `test/preset_exclude_name.py`. **CI (`.github/workflows/ci.yml`)** only runs lint, import checks, and `scripts/ci/cpu_smoke.py` (finite forward/backward, at least one non-zero grad) — no equivalence checks run in CI.

---

## 10. Weaknesses, pitfalls, and what to do differently

Verified by reading (numerically unverified because torch is unavailable here; each is easy to confirm with a 10-line script):

1. **`get_diff_weight` double-applies `scale` and drops `scalar` (LoKr and LoHa).** `get_weight` passes `scale=self.scale` into `kron_weight` (lokr.py 368 → `make_kron` 22–23), and `get_diff_weight` multiplies by `self.scale * multiplier` again (382–383). `_rebuild_forward` (563) and `bypass_forward_diff` (466, 554) apply scale once and include `scalar`; `get_diff_weight` includes neither `scalar` nor a single scale. Consequence: `merge_to`, `onfly_merge`, `parametrize`, `tools/merge.py`, and precise merges produce `ΔW·scale²` (and ignore a trained `scalar`) whenever `alpha ≠ dim` (e.g. dim=16, alpha=1 → merged strength is 1/16 of the trained forward). Invisible in the common full-matrix case (scale forced to 1) and in the test suite (which uses alpha=dim or checks nothing about magnitude). LoHa has the identical pattern (loha.py 196–235). LoCon is correct (locon.py 205–230). Present since commit `3b3122f5` (2024-05-03). A reimplementation should have exactly one place where `scale`, `scalar`, and `multiplier` are applied and test `merge_to` against the adapter forward for `alpha ≠ dim` and `use_scalar=True`.
2. **Rebuild mode materializes ΔW and runs two GEMMs per layer per step** (`_rebuild_forward`): for a 3072×12288 layer that is a 75 MB bf16 tensor per layer kept for backward, plus `(W+ΔW)−W` computed in the base weight's dtype — in bf16 any update below ~ulp(W)/2 is rounded away in the forward. For LoKr the bypass form is strictly better (never builds ΔW, `c·b·(d+a)` FLOPs), yet docs present `bypass_mode` as a bnb-only feature. Make Kronecker-aware bypass the default for Linear, or at least compute `ΔW·x` rather than `((W+ΔW)−W)·x`.
3. **Dropout semantics differ per mode:** rebuild applies `rank_dropout` (row mask on ΔW) but not `dropout`; bypass applies `dropout` on ΔW·x but not `rank_dropout`. Also `rank_dropout` in `get_weight` builds its mask on CPU with no `.to(device)` (374–378) — likely a device-mismatch error on CUDA (LoHa moves it, loha.py 224).
4. **DoRA `multiplier` semantics:** with `wd=True`, ΔW is always added at full strength and only the norm rescale is interpolated (`_weight_decompose` 137–138; lokr.py 389–392, 565–568). `multiplier=0` ≠ base model; differs from webui/Comfy loaders that scale the whole delta.
5. **`make_module_from_state_dict` for Tucker LoKr conv is wrong:** infers `lora_dim = w2a.size(1)` (265–267) and `out_dim *= w2a.size(0)` (281), but Tucker `lokr_w2_a` is stored `(r, b)` not `(b, r)` (124–126) — loading a Tucker LoKr via `create_lycoris_from_weights`/`merge` breaks unless `r == b`. Unbalanced-factorization checkpoints also cannot be rebuilt (the factor heuristic 293–315 recovers `factor=384` for a 3072 layer and then the shapes mismatch). Store the factorization explicitly (e.g. in metadata) instead of inferring it.
6. **`factor` semantics are non-obvious:** the smaller divisor always goes to W1, so `factor ≥ √dim` flips roles (`factorization(1024, 64) = (16, 64)`); non-dividing factors silently pick the largest divisor ≤ factor; prime dims collapse to W1 = 1×1. Also the docstring example table is partly wrong (`360, f=16`). Prefer explicit `(a, b)` per layer, validate divisibility, and log the result.
7. **`parametrize` swaps in/out for non-square weights** (base.py 277–279).
8. **Global class-level preset state:** `LycorisNetwork.apply_preset` mutates class attributes, keys absent from a preset are not reset, `create_lycoris` always calls it (with `"full"` by default — which for the generic wrapper only sets `ENABLE_CONV`), and `LycorisNetworkKohya` has its own parallel copy. Pass configuration objects to instances instead.
9. **Argument-name inconsistencies:** wrapper/kohya read `dora_wd` and `wd_on_output`, modules take `weight_decompose` and `wd_on_out` (so `create_lycoris(..., weight_decompose=True)` is silently ignored — `test/wrapper.py` 189 does exactly this and therefore never tests DoRA through the wrapper); `rs_lora` is only honoured by `kohya.create_network`; `decompose_both` bypasses `str_bool` so `"False"` enables it; `str_bool` treats everything except the literal `"false"` as True; `rank_dropout_scale` is not exposed at all.
10. **Functional/module drift:** functional `weight_gen` ignores `full_matrix` for Linear and creates conv `w2b` unflattened while the module flattens it; the `gamma` argument means alpha (divided by rank internally) in functional but is the final scale in the kernel-level `lokr_kron_*` — two conventions in one package.
11. **Fused kernels don't cover the popular LoKr configuration:** bypass kernel requires every factor ≤ 128, so `factor=4…8` full-matrix LoKr (W2 = 384–768 wide) falls back; the merge kernel still materializes ΔW.
12. **Kohya integration gaps:** LoRA+ only applies to `lora_up` (LoCon); `if network_module == GLoRAModule` (kohya.py 564) compares a string to a class (dead code); `train_t5xxl` is accepted but unused; DyLoRA `load_state_dict` is a no-op and its bypass is broken (also noted in `scripts/ci/cpu_smoke.py` SKIP); `FullModule.apply_to` deletes `weight` from the base module (breaks stacking/FSDP/anything reading `.weight`).
13. **Numerics:** `dora_scale` follows the module dtype (bf16 norms if the adapter is cast to bf16); `alpha` is stored pre-multiplied (`alpha·dim/r_factor`) so the file's `alpha` is not the user's alpha under `rs_lora` or full-matrix — a good compatibility trick, but document it; `custom_state_dict` returns graph-attached tensors (`lokr_w1 * scalar`) — callers must detach (the wrapper's `save_weights` does).
14. **Testing:** CI never checks forward ≡ merge, bypass ≡ rebuild, or magnitude; the wrapper test's adapter contributes exactly zero at init (zero W2 or `scalar=0`), so its forward-equality assertion cannot catch scale bugs. A reimplementation should test: bypass vs rebuild vs merged-weight forward for random non-zero factors with `alpha≠dim`, `use_scalar`, `rs_lora`, DoRA, Tucker conv, non-square Linear, and a round-trip through the saved key format.

Key files: `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/lycoris/modules/base.py`, `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/lycoris/modules/lokr.py`, `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/lycoris/functional/general.py`, `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/lycoris/functional/lokr.py`, `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/lycoris/kernels/autograd/lokr.py`, `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/lycoris/kernels/select.py`, `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/lycoris/kernels/dispatch.py`, `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/lycoris/kernels/triton/lokr/bypass.py`, `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/lycoris/wrapper.py`, `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/lycoris/kohya.py`, `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/lycoris/config.py`, `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/lycoris/config_sdk.py`, `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/lycoris/utils/__init__.py`, `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/lycoris/utils/quant.py`, `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/docs/algorithms/README.md`, `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/docs/algorithms/guidelines.md`, `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/docs/usage/network-args.md`, `/Volumes/Service/Dev/YPuddinTrainStudio/LyCORIS/example_configs/preset_configs/example.toml`.
