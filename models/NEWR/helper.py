from utils import *
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import torchvision.transforms as T
import os
import numpy as np


def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()

    # Determine model device dynamically
    device = next(model.parameters()).device

    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        # [UPDATED] Dynamically move batch to current model device
        data, train_label = [_.to(device) for _ in batch]
        logits, logits1 = model(data)
        logits = logits[:, :args.base_class]
        loss = F.cross_entropy(logits, train_label)
        logits1 = logits1[:, :args.base_class]
        loss1 = F.cross_entropy(logits1, train_label)
        acc = count_acc(logits, train_label)
        total_loss = (loss + loss1) / 2
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def test(model, testloader, epoch, args, session, result_list=None):
    test_class = args.base_class + session * args.way
    model = model.eval()
    device = next(model.parameters()).device

    vl = Averager()
    va = Averager()
    va5 = Averager()
    lgt = torch.tensor([]).to(device)
    lbs = torch.tensor([]).to(device)
    vc = Averager()
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            # [UPDATED] Cast to target device instead of forcing CPU
            data, test_label = [_.to(device) for _ in batch]
            logits, logits1 = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            top5acc = count_acc_topk(logits, test_label)
            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)
            lgt = torch.cat([lgt, logits])
            lbs = torch.cat([lbs, test_label])
        vl = vl.item()
        va = va.item()
        va5 = va5.item()
        # vc = vc.item()
        logging.info('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}'.format(epoch, vl, va, va5))
        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)
        if session > 0:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            cm = confmatrix(lgt.cpu(), lbs.cpu())  # save_model_dir
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])
            if result_list is not None:
                result_list.append(f"Seen Acc:{seenac}  Unseen Acc:{unseenac}")
            return vl, (seenac, unseenac, va)
        else:
            return vl, va


def trainNewr(model, trainloader, optimizer, scheduler, epoch, args, session, initial_weights):
    tl = Averager()  # This object is used to average the loss
    ta = Averager()  # This object is used to average the accuracy
    model = model.train()  # Set the model in training mode
    device = next(model.parameters()).device
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

    tqdm_gen = tqdm(trainloader)  # Initialize the tqdm generator for progress bar
    test_class = args.base_class + session * args.way
    mw = model.fc.weight[:test_class, :]

    for i, batch in enumerate(tqdm_gen, 1):
        data1, train_label = [_.to(device) for _ in batch]
        num_aug =5
        data = dataAug(data1, num_aug=num_aug)
        train_label = train_label.repeat(num_aug + 1)
        embed = model.encode(data)
        # Compute logits
        logits = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(mw, p=2, dim=-1)) * args.temperature
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)
        # Total loss combines classification loss and entropy terms
        total_loss = loss
        # Log the learning rate and other statistics
        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session {}, epo {}, lrc={:.4f}, total loss={:.4f}, acc={:.4f}'.format(session, epoch, lrc,
                                                                                   total_loss.item(), acc))
        # Update the average loss and accuracy
        tl.add(total_loss.item())
        ta.add(acc)
        optimizer.zero_grad()
        total_loss.backward()  # Only one backward pass here
        optimizer.step()

    model = updatedWeight(initial_weights, model, args, session)
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def testNewr(model, testloader, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device

    vl = Averager()
    va = Averager()
    va5 = Averager()
    lgt = torch.tensor([]).to(device)
    lbs = torch.tensor([]).to(device)
    mw = model.fc.weight[:test_class, :]
    mw1 = model.fc1.weight[:test_class, :]
    mw11 = mw.clone()
    mw11[:args.base_class, :] = torch.normal(0, 0.0001, (args.base_class, mw1.shape[-1]), device=device)

    with torch.no_grad():
        for i, (data, test_label) in enumerate(testloader, 1):
            data, test_label = data.to(device), test_label.to(device)

            embed = model.encode(data)
            logits = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(mw, p=2, dim=-1)) * args.temperature

            if args.NEWR==1:
                logits2 = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(mw1, p=2, dim=-1)) * args.temperature
                logits11 = F.linear(F.normalize(embed, p=2, dim=-1), F.normalize(mw11, p=2, dim=-1)) * args.temperature
                if args.taskInfo==1:
                  logits = torch.where((test_label.unsqueeze(1) < args.base_class), logits2, logits11)
                else:
                  mx1, ind1 = torch.max(logits, dim=1)
                  logits = torch.where((ind1.unsqueeze(1) < args.base_class), logits2, logits11)

            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            top5acc = count_acc_topk(logits, test_label)
            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)
            lgt = torch.cat([lgt, logits])
            lbs = torch.cat([lbs, test_label])

        vl = vl.item()
        va = va.item()
        va5 = va5.item()
        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)
        if session > 0:
            cm = confmatrix(lgt.cpu(), lbs.cpu())  # save_model_dir
            # if session ==8:
            #     #result_array = cm.detach().numpy()
            #     file_mode = "w" if i == 0 else "a"
            #     with open("confusion_matrix_cifar.csv", mode=file_mode, newline="") as file:
            #         writer = csv.writer(file)
            #         writer.writerows(cm)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[:args.base_class])
            unseenac = np.mean(perclassacc[args.base_class:])
            p = np.sum(cm[:args.base_class, :test_class])
            fn = np.sum(cm[:args.base_class, args.base_class:test_class])
            n = np.sum(cm[args.base_class:test_class, :test_class])
            fp = np.sum(cm[args.base_class:test_class, :args.base_class])
            FP = np.round(np.mean(fp / n) * 100, 2) if n > 0 else 0.0
            FN = np.round(np.mean(fn / p) * 100, 2) if p > 0 else 0.0

            logging.info(
                'epo {}, loss={:.4f} acc={:.4f}, acc@5={:.4f} '.format(getattr(args, 'epochs_new', 0), vl, va, va5))
            return vl, (seenac, unseenac, va, FN, FP)
        else:
            return vl, va


def dataAug(images, num_aug=1):
    """
    [UPDATED] Appends augmented versions of the images alongside original images.
    """
    h, w = images.shape[2], images.shape[3]
    augmentation_pipeline = T.Compose([
        T.RandomCrop((h, w), padding=w//8),  # Safe 4-pixel padding shift
        T.RandomHorizontalFlip(p=0.5),  # Safe horizontal reflection
    ])
    # Preserve original image tensor list
    augmented_images = [img for img in images]
    # Append requested number of augmented passes per image
    for _ in range(num_aug):
        for img in images:
            augmented_images.append(augmentation_pipeline(img))
    return torch.stack(augmented_images)


def updatedWeight(initial_weights, model, args, session):
    updated_weights = {k: v.clone() for k, v in model.state_dict().items()}  # Get a copy of current weights
    eta = 0.1
    alpha = 100
    new_state_dict = {}
    for (name, initial), (_, updated), layer_name in zip(initial_weights.items(), updated_weights.items(),
                                                         initial_weights):
        if args.dataset=='cifar100':
            weight_diff = updated - initial
        else:
            weight_diff = torch.abs(updated - initial)

        weight_decay_factor = torch.exp(-torch.abs(alpha * initial))
        modified_weight = initial + eta * weight_decay_factor * weight_diff
        new_state_dict[name] = modified_weight
        if session > 0:
            if layer_name == 'fc.weight':
                new_state_dict[name] = updated

    model.load_state_dict(new_state_dict)
    return model


