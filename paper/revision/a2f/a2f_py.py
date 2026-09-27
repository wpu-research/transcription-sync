"""Python port of the NVIDIA Audio2Face-3D SDK regression path (Mark v2.3)
to 52 ARKit blendshape weights, using the official open weights
(huggingface.co/nvidia/Audio2Face-3D-v2.3-Mark) and the SDK sources
(github.com/NVIDIA/Audio2Face-3D-SDK, audio2face-core).

Ported (SDK file -> here):
  - window progress (executor_regression_core.cpp GetProgressParameters):
    frame k at 30 fps has target sample k*16000/30; the network window is
    [target-4160, target+4160) (buffer_len 8320, buffer_ofs 4160), zero-padded
    outside the signal (audio_accumulator.cpp Read), scaled by input_strength
    (model_config.json, 1.3).
  - network inputs: input (B,1,8320), emotion (B,1,26) = 16 implicit emotions
    from implicit_emo_db.npz for shot g2a_neutral frame 33, then 10 explicit
    emotions (all 0, default_emotion) (model_regression.cpp).
  - output split (model_shared.cpp): [skin 272 | tongue 10 | jaw 15 | eyes 4].
  - skin PCA reconstruction (multitrack_animator.cpp): deltas = shapes_matrix_skin^T @ coefs
    (no mean added at this stage).
  - skin post-processing (multitrack_animator_cuda.cu ComputeSkinPostProcessing):
    delta = skin_strength*delta + eye_close_pose_delta*(-eyelid_open_offset + blinkOffset*blink_strength)
            + lip_open_pose_delta*lip_open_offset; two-stage exponential smoothing with
    alpha = 1-0.5^(dt/smoothing) for lower and upper face; result = shapes_mean_skin
    + upper2*upper_strength*(1-mask) + lower2*lower_strength*mask, with the face mask
    sigmoid((level - (y-ymin)/(ymax-ymin))/softness) on the neutral y coordinate.
  - blendshape solve (blendshape_solver_base.cpp Prepare, blendshape_solver.cpp):
    frontalMask vertices, active poses, A = D^T D + L1^2*0.25s*1 + L2*10s*I
    + T*100s*I + S*10s*SymM^T SymM with s = (bbox/templateBBSize)^2;
    b = D^T (target - neutral) + T*s*x_prev (exactly as in the SDK);
    bounded [0,1] minimisation of 0.5x^T A x - b^T x; cancel pairs (none active
    in this config); multipliers/offsets.
Simplified:
  - bounded QP solved exactly (scipy lsq_linear/BVLS on the Cholesky factor)
    instead of the SDK's own BVLS with tolerance; same objective and bounds.
  - blink: blinkOffset = 0 (the stochastic blink generator is not ported; it
    only affects eyelid vertices through eye_close_pose_delta).
  - tongue, jaw transform, eyes and emotion inference (Audio2Emotion) are not
    used: only skin -> 52 ARKit weights, which contains all mouth channels.
  - prediction_delay (0.25 s) appears only in a2f_ms_config.json (microservice
    config); it is not referenced anywhere in the SDK core, so frames are
    timestamped at their window target (centre) as the SDK does.
Parameter source: model_config.json (skin params) and bs_skin_config.json.
"""
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from scipy.optimize import lsq_linear

D = Path(__file__).resolve().parent
SR, FPS, BUF_LEN, BUF_OFS = 16000, 30, 8320, 4160


class A2F:
    def __init__(self, providers=('CPUExecutionProvider',), threads=4):
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(D / 'network.onnx'), so, providers=list(providers))
        mc = json.load(open(D / 'model_config.json'))['config']
        self.p = mc
        self.input_strength = mc['input_strength']
        md = np.load(D / 'model_data.npz')
        self.S = md['shapes_matrix_skin'].reshape(272, -1).astype(np.float32)      # (272, 3V)
        self.mean = md['shapes_mean_skin'].reshape(-1).astype(np.float32)
        self.lipopen = md['lip_open_pose_delta'].reshape(-1).astype(np.float32)
        self.eyeclose = md['eye_close_pose_delta'].reshape(-1).astype(np.float32)
        y = md['shapes_mean_skin'][:, 1]
        m = 1.0 / (1.0 + np.exp(-(mc['face_mask_level'] - (y - y.min()) / (y.max() - y.min())) / mc['face_mask_softness']))
        self.mask = np.repeat(m, 3).astype(np.float32)
        dt = 1.0 / FPS
        f = lambda s: 1.0 - 0.5 ** (dt / s) if s > 0 else -1.0
        self.a_low, self.a_up = f(mc['lower_face_smoothing']), f(mc['upper_face_smoothing'])
        # emotion
        e = np.load(D / 'implicit_emo_db.npz')
        names = [n.decode() for n in e['emo_spec_names']]
        i = names.index(mc['source_shot'])
        imp = e['emo_db'][e['emo_spec_start'][i] + mc['source_frame']]
        self.emotion = np.concatenate([imp, np.zeros(10, np.float32)]).astype(np.float32)[None, None, :]
        # blendshape solver
        bs = np.load(D / 'bs_skin.npz')
        self.pose_names = [n.decode() for n in bs['poseNames']][1:]
        cfg = json.load(open(D / 'bs_skin_config.json'))['blendshape_params']
        neutral = bs['neutral'].reshape(-1)
        fm = bs['frontalMask']
        idx = np.stack([3 * fm, 3 * fm + 1, 3 * fm + 2], 1).reshape(-1)
        self.act = [k for k, a in enumerate(cfg['bsSolveActivePoses']) if a]
        Dm = np.stack([bs[self.pose_names[k]].reshape(-1)[idx] for k in self.act], 1).astype(np.float64)
        self.Dm, self.idx, self.neutral_m = Dm, idx, neutral[idx].astype(np.float64)
        nv = bs['neutral']
        scale = (np.linalg.norm(nv.max(0) - nv.min(0)) / cfg['templateBBSize']) ** 2
        n = len(self.act)
        sym = cfg['bsSolveSymmetryPoses']; act_sym = [sym[k] for k in self.act]
        pairs = {}
        for j, s in enumerate(act_sym):
            pairs.setdefault(s, []).append(j)
        Sm = np.zeros((0, n))
        for s, js in pairs.items():
            if s != -1 and len(js) == 2:
                r = np.zeros((1, n)); r[0, js[0]] = 1; r[0, js[1]] = -1; Sm = np.vstack([Sm, r])
        L1, L2, Tr, Sy = cfg['strengthL1regularization'], cfg['strengthL2regularization'], cfg['strengthTemporalSmoothing'], cfg['strengthSymmetry']
        self.A = (Dm.T @ Dm + L1 * L1 * 0.25 * scale * np.ones((n, n)) + L2 * 10 * scale * np.eye(n)
                  + Tr * 100 * scale * np.eye(n) + Sy * 10 * scale * Sm.T @ Sm)
        self.Tr_s = Tr * scale
        self.R = np.linalg.cholesky(self.A).T          # A = R^T R
        self.mult = np.array(cfg['bsWeightMultipliers']); self.offs = np.array(cfg['bsWeightOffsets'])
        cancel = [cfg['bsSolveCancelPoses'][k] for k in self.act]
        self.has_cancel = any(sum(1 for c in cancel if c == v) == 2 for v in set(cancel) if v != -1)

    def network(self, audio):
        nF = int(np.floor(len(audio) / SR * FPS)) + 1
        pad = np.concatenate([np.zeros(BUF_OFS, np.float32), audio.astype(np.float32), np.zeros(BUF_LEN, np.float32)])
        win = np.stack([pad[int(round(k * SR / FPS)):int(round(k * SR / FPS)) + BUF_LEN] for k in range(nF)]) * self.input_strength
        out = []
        for s in range(0, nF, 32):
            x = win[s:s + 32][:, None, :]
            e = np.repeat(self.emotion, len(x), 0)
            out.append(self.sess.run(['result'], {'input': x, 'emotion': e})[0][:, 0, :])
        return np.vstack(out)

    def run(self, audio):
        R = self.network(audio)
        deltas = R[:, :272] @ self.S                      # (T, 3V)
        p = self.p
        base = (p['skin_strength'] * deltas + self.eyeclose[None] * (-p['eyelid_open_offset'])
                + self.lipopen[None] * p['lip_open_offset'])
        T = len(base); out = np.zeros((T, 52), np.float32)
        l1 = l2 = u1 = u2 = None
        x_prev = np.zeros(len(self.act))
        Rinv_T = np.linalg.inv(self.R.T)
        for t in range(T):
            d = base[t]
            if t == 0 or self.a_low <= 0:
                l1 = d.copy(); l2 = d.copy()
            else:
                l1 += (d - l1) * self.a_low; l2 += (l1 - l2) * self.a_low
            if t == 0 or self.a_up <= 0:
                u1 = d.copy(); u2 = d.copy()
            else:
                u1 += (d - u1) * self.a_up; u2 += (u1 - u2) * self.a_up
            geom = self.mean + u2 * p['upper_face_strength'] * (1 - self.mask) + l2 * p['lower_face_strength'] * self.mask
            b = self.Dm.T @ (geom[self.idx] - self.neutral_m) + self.Tr_s * x_prev
            # min 0.5 x^T A x - b^T x  <=>  min ||R x - R^{-T} b||^2
            x = lsq_linear(self.R, Rinv_T @ b, bounds=(0.0, 1.0), method='bvls').x
            x_prev = x
            w = np.zeros(52); w[self.act] = x
            out[t] = w * self.mult + self.offs
        return out


if __name__ == '__main__':
    import sys, soundfile as sf, time
    m = A2F()
    for p in sys.argv[1:]:
        a, sr = sf.read(p, dtype='float32'); assert sr == SR
        t0 = time.time(); W = m.run(a); dt = time.time() - t0
        j = m.pose_names.index('jawOpen')
        print(p, W.shape, f'{dt:.2f}s', 'jawOpen max %.3f mean %.3f first5 %s' % (W[:, j].max(), W[:, j].mean(), np.round(W[:5, j], 3)))
