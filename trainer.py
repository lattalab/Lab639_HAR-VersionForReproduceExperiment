import torch
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
import os
import copy

from models.cnn2d_transformer.cnn2d_transformer import CNN2D_Transformer
from dataloader.lab639_dataloader import Lab639DataLoader

import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr
        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)

class Lab639Trainer():
    def __init__(self, config):
        # config
        self.config = config

        # dataloader
        if config.mode == 'train':
            train_data_gen = Lab639DataLoader(config, split='train', seed=config.seed)
            self.train_dataloader = data.DataLoader(
                train_data_gen,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers
            )
            val_data_gen = Lab639DataLoader(config, split='val', seed=config.seed)
            self.val_dataloader = data.DataLoader(
                val_data_gen,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers
            )
        
            print(f"Samples in train set: {len(train_data_gen)}")
            print(f"Samples in val set: {len(val_data_gen)}")

        elif config.mode == 'test':
            test_data_gen = Lab639DataLoader(config, split='test', seed=config.seed)
            self.test_dataloader = data.DataLoader(
                test_data_gen,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers
            )

            print(f"Samples in test set: {len(test_data_gen)}")

        # model
        self.model = CNN2D_Transformer(
            video_encoder="vit_small_patch32_224.augreg_in21k_ft_in1k",
            len_feature=384,
            num_classes=config.num_classes,
            num_frames=config.num_frames,
            fusion_type=config.fusion_type,
            config=config
        )

        self.model = self.model.cuda()

        self.action_criterion = torch.nn.CrossEntropyLoss()
        self.view_criterion = torch.nn.CrossEntropyLoss()

        # [TESTED]
        self.tripletLoss = torch.nn.TripletMarginLoss()


        if config.mode == 'train':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate,
                                            betas=(0.9, 0.999), weight_decay=0.0005)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

            self.ema_model = copy.deepcopy(self.model)
            self.ema_optimizer = WeightEMA(self.model, self.ema_model, self.config.learning_rate, alpha=0.999)

    def train(self, writer):
        self.model.train()

        best_acc = 0
        accum_step = 16

        for epoch in range(self.config.num_epochs):
            act_acc = []
            view_acc_list = []
            action_losses = []
            view_losses = []
            sa_rgb_losses = []
            action_cl_losses = []
            view_cl_losses = []
            ortho_sub_losses = []
            ortho_act_losses = []
            self.model.train()
            for i, (video, action_label, view_labels, diff_action_video, same_action_video) in enumerate(tqdm(self.train_dataloader)):
                video = video.cuda()
                action_label = action_label.cuda()
                view_labels = view_labels.cuda()
                diff_action_video = diff_action_video.cuda()
                same_action_video = same_action_video.cuda()

                # Forward pass
                x_cls, pred_views, x_rgb, action_token, view_token, v_feat = self.model(video, is_training=True)

                with torch.no_grad():
                    # diff action features
                    diff_x_cls, _, diff_action_rgb, diff_action_token, diff_view_token, diff_v_feat = self.model(diff_action_video, is_training=True)

                    # same action features
                    same_x_cls, _, same_action_rgb, same_action_token, same_view_token, same_v_feat = self.model(same_action_video, is_training=True)

                # Compute loss
                action_loss = self.action_criterion(x_cls, action_label)
                view_loss = self.view_criterion(pred_views.view(-1, self.config.num_views), view_labels.view(-1))

                # [TESTED] modify to triplet loss

                # === Triplet Loss (SA-DV & DA-SV) ===
                B, V, D = v_feat.shape
                anchors, positives, negatives = [], [], []
                
                for v in range(V):
                    anchors.append(v_feat[:, v, :])           # Anchor: target_video, view v
                    negatives.append(diff_v_feat[:, v, :])    # DA-SV: diff_video, view v
                    v_diff = (v + 1) % V 
                    positives.append(same_v_feat[:, v_diff, :]) # SA-DV: same_video, view v_diff
                
                # Stack all views into batch dimension
                anchors = torch.cat(anchors, dim=0)
                positives = torch.cat(positives, dim=0)
                negatives = torch.cat(negatives, dim=0)
                # sa_rgb_loss = weighted_contrastive_loss(x_rgb, same_action_rgb, diff_action_rgb, action_label, same_x_cls, diff_x_cls)
                sa_rgb_loss = self.tripletLoss(anchors, positives, negatives)

                subqloss = 0 
                actqloss = 0    
                            
                for subq in view_token:
                    dist = abs(cosine_pairwise_dist(subq, subq))
                    subqloss += torch.sum(dist - torch.eye(subq.shape[0]).cuda())
                
                for actq in action_token:
                    dist = abs(cosine_pairwise_dist(actq, actq))
                    actqloss += torch.sum(dist - torch.eye(actq.shape[0]).cuda())

                # [TEST] : Modify loss
                # if epoch < 25:
                #     cl_lambda = 0
                # else :
                #     cl_lambda = 1
                # # cl_lambda = 1
                # cl_lambda = get_contrastive_lambda(epoch + 1, warmup_epochs = 0, max_epoch = 30)
                # loss = action_loss + view_loss + subqloss + actqloss + cl_lambda * (sa_rgb_loss)
                # # loss = loss / accum_step
                # loss.backward()
                loss = action_loss + view_loss + subqloss + actqloss + sa_rgb_loss
                loss.backward()

                # if (i + 1) % accum_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.ema_optimizer.step()

                x_cls = torch.argmax(x_cls, dim=1)
                action_losses.append(action_loss.item())
                acc_tmp = torch.sum(x_cls == action_label)
                act_acc.append(acc_tmp)

                pred_view_label = torch.argmax(pred_views, dim=-1)  # [B, V]
                correct_view = (pred_view_label == view_labels).sum().item()
                total_view = view_labels.numel()
                view_acc = correct_view / total_view
                view_acc_list.append(view_acc)
                view_losses.append(view_loss.item())

                ortho_sub_losses.append(subqloss.item())
                ortho_act_losses.append(actqloss.item())

                sa_rgb_losses.append(sa_rgb_loss.item())

            # Compute accuracy
            total_correct = torch.sum(torch.stack(act_acc)).item()
            total_samples = len(self.train_dataloader.dataset)
            acc_epoch = total_correct / total_samples
            print(f"Epoch [{epoch + 1}/{self.config.num_epochs}], Action Loss: {np.mean(action_losses):.4f}, Action Accuracy: {acc_epoch * 100:.2f}%")

            # Compute view accuracy
            view_acc_epoch = np.mean(view_acc_list)
            print(f"Epoch [{epoch + 1}/{self.config.num_epochs}], View Loss: {np.mean(view_losses):.4f}, View Accuracy: {view_acc_epoch * 100:.2f}%")

            # Compute sa_rgb_loss and sa_flow_loss
            print(f"Epoch [{epoch + 1}/{self.config.num_epochs}], sa_rgb_loss: {np.mean(sa_rgb_losses):.4f}")

            ortho_sub_loss = np.mean(ortho_sub_losses)
            ortho_act_loss = np.mean(ortho_act_losses)
            print(f"Epoch [{epoch + 1}/{self.config.num_epochs}], ortho_sub_loss: {ortho_sub_loss:.4f}, ortho_act_loss: {ortho_act_loss:.4f}")

            writer.add_scalar('train/action loss', np.mean(action_losses), epoch)
            writer.add_scalar('train/action accuracy', acc_epoch * 100, epoch)
            writer.add_scalar('train/view loss', np.mean(view_losses), epoch)
            writer.add_scalar('train/view accuracy', view_acc_epoch * 100, epoch)
            writer.add_scalar('train/ortho_sub loss', ortho_sub_loss, epoch)
            writer.add_scalar('train/ortho_act loss', ortho_act_loss, epoch)
            writer.add_scalar('train/sa_rgb loss', np.mean(sa_rgb_losses), epoch)

            # Validation
            if (epoch) % self.config.validation_interval == 0:
                avg_loss1, action_accuracy1, view_accuracy1, class_correct1, class_total1 = self.validate(epoch, self.model)
                avg_loss2, action_accuracy2, view_accuracy2, class_correct2, class_total2 = self.validate(epoch, self.ema_model)

                if action_accuracy1 > action_accuracy2:
                    avg_loss, action_accuracy, view_accuracy = avg_loss1, action_accuracy1, view_accuracy1
                    class_correct, class_total = class_correct1, class_total1
                else:
                    avg_loss, action_accuracy, view_accuracy = avg_loss2, action_accuracy2, view_accuracy2
                    class_correct, class_total = class_correct2, class_total2

                writer.add_scalar('val/loss', avg_loss, epoch)
                writer.add_scalar('val/action accuracy', action_accuracy, epoch)
                writer.add_scalar('val/view accuracy', view_accuracy, epoch)

                for i in range(self.config.num_classes):
                    writer.add_scalar(f'class_val_accuracy/class_{i}_accuracy', class_correct[i] / class_total[i], epoch)

                # Save model
                if action_accuracy > best_acc:
                    best_acc = action_accuracy
                    save_path = os.path.join(self.config.result_path, 'pth', self.config.output_name)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    # remove old models
                    for file in os.listdir(save_path):
                        if file.endswith('.pth'):
                            os.remove(os.path.join(save_path, file))
                    save_state_dict = self.model.state_dict() if action_accuracy1 > action_accuracy2 else self.ema_model.state_dict()
                    torch.save(save_state_dict, os.path.join(save_path, 'model_{}_{:.4f}.pth'.format(epoch, action_accuracy)))
                    print(f"Model saved to {os.path.join(save_path, 'model_{}_{:.4f}.pth'.format(epoch, action_accuracy))}, Action Accuracy: {action_accuracy:.2f}%")

        print("Training complete.")

    def validate(self, epoch, model):
        model.eval()
        total_loss = 0
        action_correct = 0
        view_correct = 0

        class_correct = list(0. for i in range(self.config.num_classes))
        class_total = list(0. for i in range(self.config.num_classes))

        with torch.no_grad():
            for i, (video, action_label, view_labels, keys) in enumerate(self.val_dataloader):
                video = video.cuda()
                action_label = action_label.cuda()
                view_labels = view_labels.cuda()

                # Forward pass
                # [TEST] add extra return value
                x_cls, pred_views, x_rgb, _, _, _ = model(video, is_training=False)

                # Compute loss
                action_loss = self.action_criterion(x_cls, action_label)
                view_loss = self.view_criterion(pred_views.view(-1, self.config.num_views), view_labels.view(-1))

                loss = action_loss + view_loss
                total_loss += loss.item()

                # Compute action_accuracy
                _, predicted = torch.max(x_cls.data, 1)
                action_correct += (predicted == action_label).sum().item()

                pred_view_label = torch.argmax(pred_views, dim=-1)  # [B, V]
                view_correct += (pred_view_label == view_labels).sum().item()

                for j in range(len(action_label)):
                    label = int(action_label[j].item())
                    pred = int(predicted[j].item())
                    if label == pred:
                        class_correct[label] += 1
                    class_total[label] += 1

        avg_loss = total_loss / len(self.val_dataloader)
        action_accuracy = 100 * action_correct / len(self.val_dataloader.dataset)
        view_accuracy = 100 * view_correct / (len(self.val_dataloader.dataset) * self.config.num_views)
        print(f"Validation - Epoch [{epoch + 1}/{self.config.num_epochs}], Loss: {avg_loss:.4f}, Action Accuracy: {action_accuracy:.2f}%, View Accuracy: {view_accuracy:.2f}%")
        print("Validation complete.")

        return avg_loss, action_accuracy, view_accuracy, class_correct, class_total

    def test(self, pth_path):
        self.model.load_state_dict(torch.load(pth_path))
        self.model.eval()
        total_loss = 0
        action_correct = 0
        view_correct = 0

        num_classes = self.config.num_classes

        fold_action_preds = []
        fold_action_labels = []
        miss_keys = []
        miss_pred = []

        class_correct = list(0. for i in range(self.config.num_classes))
        class_total = list(0. for i in range(self.config.num_classes))

        with torch.no_grad():
            for i, (video, action_label, view_labels, keys) in enumerate(tqdm(self.test_dataloader)):
                video = video.cuda()
                action_label = action_label.cuda()
                view_labels = view_labels.cuda()

                # Forward pass
                # [TEST] add extra return value
                x_cls, pred_views, x_rgb, _, _, _ = self.model(video, is_training=False)

                # Compute loss
                action_loss = self.action_criterion(x_cls, action_label)
                view_loss = self.view_criterion(pred_views.view(-1, self.config.num_views), view_labels.view(-1))

                loss = action_loss + view_loss
                total_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(x_cls.data, 1)
                action_correct += (predicted == action_label).sum().item()

                fold_action_preds.extend(predicted.cpu().numpy())
                fold_action_labels.extend(action_label.cpu().numpy())

                x_cls = torch.argmax(x_cls, dim=1)

                for j in range(len(action_label)):
                    label = int(action_label[j].item())
                    pred = int(x_cls[j].item())
                    if label == pred:
                        class_correct[label] += 1
                    else:
                        miss_keys.append(keys[j])
                        miss_pred.append(pred + 1)
                    class_total[label] += 1

        avg_loss = total_loss / len(self.test_dataloader)
        accuracy = 100 * action_correct / len(self.test_dataloader.dataset)
        print(f"Test - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # print all class's accuracy
        for i in range(self.config.num_classes):
            print('Test Accuracy of class {}: {:.4f}'.format(i, class_correct[i] / class_total[i]), flush=True)

        return avg_loss, accuracy, fold_action_preds, fold_action_labels, miss_keys, miss_pred

def cosine_pairwise_dist(x, y):
    assert x.shape[1] == y.shape[1], "both sets of features must have same shape"
    return torch.nn.functional.cosine_similarity(x[:, None, :], y[None, :, :], dim=-1)

def weighted_contrastive_loss(anchor, positive, negative, target, pos_logits, neg_logits):
    pos_weights = torch.softmax(pos_logits, dim=1).gather(1, target.view(-1, 1)).squeeze(1)
    neg_weights = torch.exp(torch.softmax(neg_logits, dim=1).gather(1, target.view(-1, 1)).squeeze(1))
    sim_pos = torch.nn.functional.cosine_similarity(anchor, positive)
    sim_neg = torch.nn.functional.cosine_similarity(anchor.unsqueeze(1), negative.unsqueeze(0), dim=2)  # pairwise
    sim_neg = sim_neg * neg_weights.unsqueeze(0)
    denom = torch.exp(sim_pos / 0.1) + torch.sum(torch.exp(sim_neg / 0.1), dim=1)
    loss = -torch.log(torch.exp(sim_pos / 0.1) / denom)

    return loss.mean()


def get_contrastive_lambda(epoch, max_lambda=1, warmup_epochs=0, max_epoch=100):
    if epoch < warmup_epochs:
        return 0.0
    else:
        progress = (epoch - warmup_epochs) / (max_epoch - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        scale = 12
        # sigmoid from 0 to 1
        ramp = 1 / (1 + np.exp(-scale * (progress - 0.5)))
        return max_lambda * ramp
    
    return loss.mean()
