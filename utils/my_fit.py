import torch
from tqdm import tqdm
from utils.utils import get_lr
import os

def fit_one_epoch(model_train, model, yolo_loss, loss_history,
                  optimizer, epoch, epoch_step, epoch_step_val,
                  gen, gen_val, Epoch, cuda, save_period, save_dir):
    loss = 0
    val_loss = 0
    model_train.train()

    print('Start Train')
    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    for iteration, batch in enumerate(gen):
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]

        optimizer.zero_grad()

        outputs = model_train(images)
        loss_value_all = 0
        # ----------------------#
        #   计算损失
        # ----------------------#
        for l in range(len(outputs)):
            loss_item = yolo_loss(l, outputs[l], targets)
            loss_value_all += loss_item
        loss_value = loss_value_all

        # ----------------------#
        #   反向传播
        # ----------------------#
        loss_value.backward()
        optimizer.step()

        loss += loss_value.item()

        pbar.set_postfix(**{'loss': loss / (iteration + 1),
                            'lr': get_lr(optimizer)})
        pbar.update(1)
    pbar.close()
    print('Finish Train')
    print('Start Validation')

    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(images)

            loss_value_all = 0
            # ----------------------#
            #   计算损失
            # ----------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all += loss_item
            loss_value = loss_value_all

        val_loss += loss_value.item()
        pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
        pbar.update(1)

    pbar.close()
    print('Finish Validation')
    loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

    # -----------------------------------------------#
    #   保存权值
    # -----------------------------------------------#
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
        epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

