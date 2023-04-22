#!/usr/bin/env python3

from .training_helper import AverageMeter
from .config import *
from .common_helper import make_model, make_outdir

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
from time import time
import torch


def get_infer_dict(model, loader):
    model.eval()
    infer_dict = {}

    start_time = time.time()

    with torch.no_grad():
        ability_mus, item_feat_mus = [], []
        ability_logvars, item_feat_logvars = [], []

        pbar = tqdm(total=len(loader))
        for _, response, _, mask in loader:
            mb = response.size(0)
            response = response.to(device)
            mask = mask.long().to(device)

            _, ability_mu, ability_logvar, _, item_feat_mu, item_feat_logvar = \
                model.encode(response, mask)

            ability_mus.append(ability_mu.cpu())
            ability_logvars.append(ability_logvar.cpu())

            item_feat_mus.append(item_feat_mu.cpu())
            item_feat_logvars.append(item_feat_logvar.cpu())

            pbar.update()

        ability_mus = torch.cat(ability_mus, dim=0)
        ability_logvars = torch.cat(ability_logvars, dim=0)
        pbar.close()

    infer_dict['ability_mu'] = ability_mus
    infer_dict['ability_logvar'] = ability_logvars
    infer_dict['item_feat_mu'] = item_feat_mu
    infer_dict['item_feat_logvar'] = item_feat_logvar

    return infer_dict


def sample_posterior_predictive(model, loader):
    model.eval()
    meter = AverageMeter()
    pbar = tqdm(total=len(loader))

    start_time = time.time()

    with torch.no_grad():

        response_sample_set = []

        for missing_label, response, missing_mask, mask in loader:
            mb = response.size(0)
            response = response.to(device)
            mask = mask.long().to(device)

            _, ability_mu, ability_logvar, _, item_feat_mu, item_feat_logvar = \
                model.encode(response, mask)

            ability_scale = torch.exp(0.5 * ability_logvar)
            item_feat_scale = torch.exp(0.5 * item_feat_logvar)

            ability_posterior = torch.distributions.Normal(ability_mu, ability_scale)
            item_feat_posterior = torch.distributions.Normal(item_feat_mu, item_feat_scale)

            ability_samples = ability_posterior.sample([num_posterior_samples])
            item_feat_samples = item_feat_posterior.sample([num_posterior_samples])

            response_samples = []
            for i in range(num_posterior_samples):
                ability_i = ability_samples[i]
                item_feat_i = item_feat_samples[i]
                response_i = model.decode(ability_i, item_feat_i).cpu()
                response_samples.append(response_i)
            response_samples = torch.stack(response_samples)
            response_sample_set.append(response_samples)

            pbar.update()

        response_sample_set = torch.cat(response_sample_set, dim=1)

        pbar.close()

    end_time = time.time()

    return {'response': response_sample_set, 'infer_time': end_time - start_time}


def sample_posterior_mean(model, loader):
    model.eval()
    meter = AverageMeter()
    pbar = tqdm(total=len(loader))

    start_time = time.time()

    with torch.no_grad():
        response_sample_set = []
        correct_hidden = []
        for missing_label, response, missing_mask, mask in loader:
            mb = response.size(0)
            response = response.to(device)
            mask = mask.long().to(device)

            _, ability_mu, _, _, item_feat_mu, _ = \
                model.encode(response, mask)

            response_sample = model.decode(ability_mu, item_feat_mu).cpu()
            indices = missing_mask.to_numpy().nonzero()
            correct_hidden.append(missing_label[indices])
            response_sample_set.append(response_sample[indices])
            pbar.update()

        pbar.close()

    end_time = time.time()

    return {'response': response_sample_set,
            'infer_time': end_time - start_time,
            'correct': correct_hidden}


def run_inference_new_loader(model, dataset, out_dir):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
    )
    for checkpoint_name in ['model_best.pth.tar']:
        checkpoint = torch.load(os.path.join(out_dir, checkpoint_name))
        model.load_state_dict(checkpoint['model_state_dict'])
        posterior_mean_samples = sample_posterior_mean(loader)
        checkpoint['posterior_mean_samples'] = posterior_mean_samples
        y_pred_prob = posterior_mean_samples['response'].numpy()
        y_pred = np.round(y_pred_prob)
        y_gt = posterior_mean_samples['correct'].numpy()

        acc = accuracy_score(y_gt, y_pred)
        print(f'Missing Imputation Accuracy from samples: {acc}')
        auc = roc_auc_score(y_gt, y_pred_prob)
        print("AUC: {:.2f}".format(auc))

        cm = confusion_matrix(y_gt, y_pred)
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot()

        print(f'Infer time:', posterior_mean_samples['infer_time'])


def run_inference(model, dataset):
    dataset = train_dataset if train else test_dataset
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = False,
    )

    for checkpoint_name in ['model_best.pth.tar']:
        checkpoint = torch.load(os.path.join(out_dir, checkpoint_name))
        model.load_state_dict(checkpoint['model_state_dict'])
        posterior_mean_samples = sample_posterior_mean(loader)
        checkpoint['posterior_mean_samples'] = posterior_mean_samples

        if artificial_missing_perc > 0:
            missing_indices = dataset.missing_indices
            missing_labels = dataset.missing_labels

            if np.ndim(missing_labels) == 1:
                missing_labels = np.round(missing_labels[:, np.newaxis])

            y_pred_prob = posterior_mean_samples['response'].squeeze(0).squeeze(2).numpy()
            y_pred = np.round(y_pred_prob)

            correct, count = 0, 0
            predicted_labels = []
            predicted_label_probs = []
            for missing_index, missing_label in zip(missing_indices, missing_labels):
                predicted_label = y_pred[missing_index[0], missing_index[1]]
                if predicted_label.item() == missing_label[0]:
                    correct += 1
                count += 1
                predicted_labels.append(predicted_label.item())
                predicted_label_probs.append(y_pred_prob[missing_index[0], missing_index[1]])

            print('y_pred:', np.unique(predicted_labels, return_counts=True))
            print('y:', np.unique(missing_labels.flatten(), return_counts=True))

            missing_imputation_accuracy = correct / float(count)
            checkpoint['missing_imputation_accuracy'] = missing_imputation_accuracy
            print(f'Missing Imputation Accuracy from samples: {missing_imputation_accuracy}')

            auc = roc_auc_score(missing_labels.flatten(), predicted_label_probs)
            print("AUC: {:.2f}".format(auc))

            cm = confusion_matrix(missing_labels.flatten(), predicted_labels)
            print(cm)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
            disp.plot()

            print(f'Infer time:', posterior_mean_samples['infer_time'])
        torch.save(checkpoint, os.path.join(out_dir, checkpoint_name))


def main():
    out_dir = make_outdir()
    train_loader, test_loader = make_dataloaders()
    model = make_model()
    run_inference_new_loader(model, test_loader, out_dir)


if __name__ == '__main__':
    main()
