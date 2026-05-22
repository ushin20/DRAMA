import argparse
import torch
import os
import numpy as np

from torch.utils.data import DataLoader
from deco import ProgressBar
from model import SlidingDataset, TestLoss, DRAMA
from utils import build_spatial_edge
from eval import calc_result, calc_seq

def search_best_weights(loss_norm, drift_norm, true_labels, target_metrics='point-adjusted', mode='water'):
    print(f"Target metrics is '{target_metrics}'.")
    best_f1, best_alpha = -1, None

    for alpha in np.linspace(0, 1, 11):
        alpha = round(alpha,1)
        beta = 1 - alpha
        combined = alpha * loss_norm + beta * drift_norm  # [T,C]
        if mode == 'water':
            scores = np.max(combined, axis=1)
        elif mode == 'space':
            scores = combined

        result, _ = calc_result(scores, true_labels, mode=target_metrics, verbose=False)
        f1, precision, recall, tp, tn, fp, fn = result
        if target_metrics == 'composite':
            msg = f"Test alpha={alpha:.1f} F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, TP_time: {tp}, FP_time: {tn}, TP_event: {fp}, FN_event: {fn}"
        else:
            msg = f"Test alpha={alpha:.1f} F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}"
        
        if mode != 'space':
            print(msg)
            
        if (f1 > best_f1):
            best_f1, best_alpha = f1, alpha

    return best_alpha, 1 - best_alpha, best_f1

def zscore_normalize(x):
    ''' channel-wise normalziation '''
    mean = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True)

    return (x - mean) / (std + 1e-6)

@torch.no_grad()
def run_test(model: DRAMA, loader, loss_fn, device, mode='water'):
    model.eval()
    model.reset_mem()
    all_loss = []
    all_drift = []
    all_label = []

    bar = ProgressBar(loader, total=len(loader), unit='batch', desc=f"Testing")
    for x, label in bar:
        x = x.to(device)
        x_hat = model(x)
        if mode == 'water':
            loss = loss_fn(x_hat, x) #[b,c]
        elif mode == 'space':
            loss = torch.sqrt(torch.mean((x[:,0,:]-x_hat[:,0,:])**2, dim=(1))) # [B]        

        drift = model.get_drift_score(per_sensor=True, mode='cosine') #[B,C]
    
        if mode == 'space':
            drift = drift[:,0]

        all_loss.append(loss.cpu()) 
        all_drift.append(drift.cpu())
        all_label.append(label.cpu())

    all_loss = torch.cat(all_loss, dim=0)
    all_drift = torch.cat(all_drift, dim=0)
    all_label = torch.cat(all_label, dim=0)
    
    all_loss = all_loss.cpu().numpy()
    all_drift = all_drift.cpu().numpy()
    all_label = all_label.cpu().numpy()
    
    return all_loss, all_drift, all_label

def test(config):
    print("-"*100)
    print(f"Testing '{config.dataset}-{config.tag}'")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = (
        f"models/{config.dataset}/"
        f"{config.tag}_W{config.W}_D{config.D}_M{config.M}_H{config.H}_K{config.K}_decay{config.decay:.3f}_B{config.B}.pth"
    )
    result_path = (
        f"results/{config.dataset}/"
        f"{config.tag}_W{config.W}_D{config.D}_M{config.M}_H{config.H}_K{config.K}_decay{config.decay:.3f}_B{config.B}/"
    )
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    assert os.path.exists(model_path), f"Model not found at {model_path}"


    if config.pretested:
        loss_path = os.path.join(result_path, 'loss.npy')
        if not os.path.exists(loss_path):
            raise FileNotFoundError(f"Pretested result not found at {loss_path}. Please run test without '-pretested' flag first.")
        loss = np.load(loss_path)
        drift = np.load(os.path.join(result_path, 'drift.npy'))
        label = np.load(os.path.join(result_path, 'label.npy'))
    else:
        print("Loading test dataset...")
        
        # ----------------------------------------------------------------------------
        # Load test data
        x_test = np.load(f'./data/{config.dataset}/preprocessed/test.npy')
        y_label = np.load(f'./data/{config.dataset}/preprocessed/label.npy')
        # ----------------------------------------------------------------------------
        
        print("Test:", x_test.shape)
        test_dataset = SlidingDataset(x_test, labels=y_label, window_size=config.W)
        test_loader = DataLoader(test_dataset, batch_size=config.B, shuffle=False, pin_memory=True, drop_last=True)

        # -- Load model --
        print("Initializing model...")
        c = x_test.shape[1]
        edge = build_spatial_edge(mode=config.dataset)

        model = DRAMA(
            b=config.B,
            d=config.D,
            c=c,
            m=config.M,
            w=config.W,
            h=config.H,
            decay=config.decay, E_spatial=edge, topk=config.K,
            device=device
        ).to(device)
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        # -- Run test --
        loss_fn = TestLoss()
        loss, drift, label = run_test(model, test_loader, loss_fn, device, mode='water')

        # -- Save --
        np.save(os.path.join(result_path, 'loss.npy'), loss)
        np.save(os.path.join(result_path, 'drift.npy'), drift)
        np.save(os.path.join(result_path, 'label.npy'), label)

    # Evaluate
    loss = zscore_normalize(loss)
    drift = zscore_normalize(drift)

    best_alpha = search_best_weights(loss, drift, label, target_metrics=config.search_metric)[0]
    print("Best alpha=", best_alpha)
    score = best_alpha*loss+(1-best_alpha)*drift
    anomaly_score = np.max(score, axis=1)

    results = {}
    for mode in ['point-wise', 'point-adjusted', 'composite']:
        result, threshold = calc_result(anomaly_score, label, mode=mode, verbose=False)
        results[mode] = result

        if mode == 'composite':
            f1, p, r, tp_t, fp_t, tp_e, fn_e = result
            print(f"{mode.upper():<16} | P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f} | TP_t: {tp_t} | FP_t: {fp_t} | TP_e: {tp_e} | FN_e: {fn_e} | Th: {threshold:.6f}")
            f1, p, r, tp_t, fp_t, tp_e, fn_e = calc_seq(anomaly_score, label, threshold, mode='point-adjusted')
            print(f"With this setting, PA=> P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f} | TP: {tp_t} | TN: {fp_t} | FP: {tp_e} | FN: {fn_e} | Th: {threshold:.6f}")
            results['final'] = (f1, p, r, tp_t, fp_t, tp_e, fn_e)
        else:
            f1, p, r, tp, tn, fp, fn = result
            print(f"{mode.upper():<16} | P: {p:.4f} | R: {r:.4f} | F1: {f1:.4f} | TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn} | Th: {threshold:.6f}")

    return results

def testNSL(config):
    print("-"*100)
    print(f"Testing NSL '{config.dataset}-{config.tag}'")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nsl_projs = sorted(set(
        x.split("_")[0]
        for x in os.listdir(f'./data/{config.dataset}/')
        if '_test.npy' in x
    ))

    nsl_f1, nsl_pre, nsl_rec = 0,0,0
    nsl_TP, nsl_TN, nsl_FP, nsl_FN = 0,0,0,0
    skipped = 0
    for it, proj in enumerate(nsl_projs):
        if proj == 'P-2':
            skipped += 1
            continue
                
        print("-"*100)
        print(f"[{it}/{len(nsl_projs)}] '{config.dataset}-{proj}-{config.tag}'")
        model_path = (
            f"models/{config.dataset}/"
            f"{proj}_{config.tag}_W{config.W}_D{config.D}_M{config.M}_H{config.H}_K{config.K}_decay{config.decay:.3f}_B{config.B}.pth"
        )
        result_path = (
            f'results/{config.dataset}/'
            f'{config.tag}_W{config.W}_D{config.D}_M{config.M}_H{config.H}_K{config.K}_decay{config.decay:.3f}_B{config.B}/'
        )
        os.makedirs(result_path, exist_ok=True)

        assert os.path.exists(model_path), f"Model not found at {model_path}"

        if config.pretested and os.path.exists(os.path.join(result_path, f'{proj}_loss.npy')):
            loss = np.load(os.path.join(result_path, f'{proj}_loss.npy'))
            drift = np.load(os.path.join(result_path, f'{proj}_drift.npy'))
            label = np.load(os.path.join(result_path, f'{proj}_label.npy'))
        else:
            # ----------------------------------------------------------------------------
            # Load data
            x_test = np.load(f'./data/{config.dataset}/{proj}_test.npy')
            y_label = np.load(f'./data/{config.dataset}/{proj}_label.npy')
            # ----------------------------------------------------------------------------
            
            test_dataset = SlidingDataset(x_test, labels=y_label, window_size=config.W)
            test_loader = DataLoader(test_dataset, batch_size=config.B, shuffle=False, pin_memory=True, drop_last=True)
            c = x_test.shape[1]

            # -- Load model
            edge = build_spatial_edge(mode=config.dataset)
            model = DRAMA(
                b=config.B,
                d=config.D,
                c=c,
                m=config.M,
                w=config.W,
                h=config.H,
                decay=config.decay, E_spatial=edge, topk=config.K,
                device=device
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            print(f"Loaded model from {model_path}")

            # -- Run test
            loss_fn = TestLoss()
            loss, drift, label = run_test(model, test_loader, loss_fn, device, mode='space')

            # -- Save
            np.save(os.path.join(result_path, f'{proj}_loss.npy'), loss)
            np.save(os.path.join(result_path, f'{proj}_drift.npy'), drift)
            np.save(os.path.join(result_path, f'{proj}_label.npy'), label)
        
        print("-"*100)
        loss = zscore_normalize(loss)
        drift = zscore_normalize(drift)

        best_alpha = search_best_weights(loss, drift, label, target_metrics=config.search_metric, mode='space')[0]
        print("Best alpha=", best_alpha)
        score = best_alpha*loss+(1-best_alpha)*drift
        anomaly_score = score

        _, threshold = calc_result(anomaly_score, label, mode=config.search_metric, verbose=False)
        result = calc_seq(anomaly_score, label, threshold, mode=config.search_metric)
        print(f"[{proj}] {config.search_metric.upper()}:", result)

        nsl_f1 += result[0]
        nsl_pre+= result[1]
        nsl_rec+= result[2]
        nsl_TP += result[3] #fc1: tp_t
        nsl_TN += result[4] #fc1: fp_t
        nsl_FP += result[5] #fc1: tp_e
        nsl_FN += result[6] #fc1: fn_e
    
    beta = 1
    if config.search_metric == 'composite':
        nsl_p = nsl_TP / (nsl_TP + nsl_TN + 1e-6)
        nsl_r = nsl_FP / (nsl_FP + nsl_FN + 1e-6)
        nsl_f = (1+beta**2) * nsl_p * nsl_r / (beta**2*nsl_p+nsl_r)
    else:
        nsl_p = nsl_TP / (nsl_TP + nsl_FP)
        nsl_r = nsl_TP / (nsl_TP + nsl_FN)
        nsl_f = (1+beta**2) * nsl_p * nsl_r / (beta**2*nsl_p+nsl_r)

    print(f"MICRO: {config.search_metric.upper()}: "  # -- main metric
      f"F1={nsl_f:.4f}, "
      f"Precision={nsl_p:.4f}, "
      f"Recall={nsl_r:.4f}")
    
    length = len(nsl_projs)-skipped # -- for macro average, exclude P-2
    print(f"MACRO: {config.search_metric.upper()}: " 
      f"F1={nsl_f1/length:.4f}, "
      f"Precision={nsl_pre/length:.4f}, "
      f"Recall={nsl_rec/length:.4f}")
    
    return nsl_f, nsl_p, nsl_r


if __name__ == "__main__":
    # ---------------------- Config ----------------------
    parser = argparse.ArgumentParser(description="Test model on prediction task.")
    parser.add_argument("--dataset", type=str, required=True, choices=['swat', 'wadi', 'smap', 'msl'])
    parser.add_argument("--tag", type=str, required=True)

    parser.add_argument("--W", help='window size', type=int, default=5)
    parser.add_argument("--D", help='memory horizon', type=int, default=30)            
    parser.add_argument("--B", help='batch size', type=int, default=64)        
    parser.add_argument("--K", help='topk', type=int, default=5)             
    parser.add_argument("--M", help='embedding dimension', type=int, default=30)          
    parser.add_argument("--H", help='hidden dimension', type=int, default=32)            
    parser.add_argument("--decay", help='memory decay factor', type=float, default=0.05)
    parser.add_argument("--downsampling", help='downsampling', type=int, default=0)    

    # -- Test setting parameter
    parser.add_argument("--pretested", action='store_true', default=False)
    parser.add_argument("--search_metric", type=str, default='point-adjusted', choices=['point-wise', 'point-adjusted', 'composite'])
    config = parser.parse_args()
    
    if config.dataset == 'wadi':
        config.W = 30
        config.D = 30
        config.B = 64
        config.M = 30
        config.H = 32
        config.decay = 0.05
        config.downsampling = 0

    elif config.dataset == 'swat':
        config.W = 5
        config.D = 50
        config.B = 64
        config.M = 30
        config.H = 32
        config.decay = 0.05
        config.downsampling = 0

    elif config.dataset == 'smap':
        config.W = 30
        config.D = 30
        config.B = 64
        config.M = 30
        config.H = 32
        config.decay = 0.05
        config.downsampling = 0

    elif config.dataset == 'msl':
        config.W = 30
        config.D = 30
        config.B = 64
        config.M = 30
        config.H = 32
        config.decay = 0.05
        config.downsampling = 0

    # -- Test
    print(config)
    
    result_text_file = f'./results/{config.dataset}.txt'
    print("Results will be saved on", result_text_file)
    config_dict = vars(config)
    result_str = ', '.join(f"{k}:{v}" for k, v in config_dict.items())+'\n'
    if config.dataset in ['swat', 'wadi']:
        results=test(config)
        for k,v in results.items():
            if v is None or np.isscalar(v):
                if v is None:
                    result_str += f"[{k}] None\n"
                else:
                    result_str += f"[{k}] {v:.4f}\n"
                continue
            f1,pre,rec = v[:3]
            result_str += f"[{k}] F1={f1:.4f}, Pre={pre:.4f}, Rec={rec:.4f}\n"
    else:
        results=testNSL(config)
        result_str += f"[{config.search_metric}] F1={results[0]:.4f}, Pre={results[1]:.4f}, Rec={results[2]:.4f}\n"

    result_str += "-"*30 + '\n'
    with open(result_text_file, 'a') as f:
        f.write(result_str)
