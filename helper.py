import json
import math
import time
import random

import matplotlib.pyplot as plt
import seaborn
import torch
import torchvision
import cv2
import numpy as np

def hello_helper():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from a2_helper.py!')


def rel_error(x, y, eps=1e-10):
    """
    Compute the relative error between a pair of tensors x and y,
    which is defined as:

                            max_i |x_i - y_i]|
    rel_error(x, y) = -------------------------------
                      max_i |x_i| + max_i |y_i| + eps

    Inputs:
    - x, y: Tensors of the same shape
    - eps: Small positive constant for numeric stability

    Returns:
    - rel_error: Scalar giving the relative error between x and y
    """
    """ returns relative error between x and y """
    top = (x - y).abs().max().item()
    bot = (x.abs() + y.abs()).clamp(min=eps).max().item()
    return top / bot


def compute_numeric_gradient(f, x, dLdf=None, h=1e-7):
    """
    Compute the numeric gradient of f at x using a finite differences
    approximation. We use the centered difference:

    df    f(x + h) - f(x - h)
    -- ~= -------------------
    dx           2 * h

    Function can also expand this easily to intermediate layers using the
    chain rule:

    dL   df   dL
    -- = -- * --
    dx   dx   df

    Inputs:
    - f: A function that inputs a torch tensor and returns a torch scalar
    - x: A torch tensor giving the point at which to compute the gradient
    - dLdf: optional upstream gradient for intermediate layers
    - h: epsilon used in the finite difference calculation
    Returns:
    - grad: A tensor of the same shape as x giving the gradient of f at x
    """
    flat_x = x.contiguous().flatten()
    grad = torch.zeros_like(x)
    flat_grad = grad.flatten()

    # Initialize upstream gradient to be ones if not provide
    if dLdf is None:
        y = f(x)
        dLdf = torch.ones_like(y)
    dLdf = dLdf.flatten()

    # iterate over all indexes in x
    for i in range(flat_x.shape[0]):
        oldval = flat_x[i].item()  # Store the original value
        flat_x[i] = oldval + h  # Increment by h
        fxph = f(x).flatten()  # Evaluate f(x + h)
        flat_x[i] = oldval - h  # Decrement by h
        fxmh = f(x).flatten()  # Evaluate f(x - h)
        flat_x[i] = oldval  # Restore original value

        # compute the partial derivative with centered formula
        dfdxi = (fxph - fxmh) / (2 * h)

        # use chain rule to compute dLdx
        flat_grad[i] = dLdf.dot(dfdxi).item()

    # Note that since flat_grad was only a reference to grad,
    # we can just return the object in the shape of x by returning grad
    return grad


def reset_seed(number):
    """
    Reset random seed to the specific number

    Inputs:
    - number: A seed number to use
    """
    random.seed(number)
    torch.manual_seed(number)
    return


def tensor_to_image(tensor):
    """
    Convert a torch tensor into a numpy ndarray for visualization.

    Inputs:
    - tensor: A torch tensor of shape (3, H, W) with
      elements in the range [0, 1]

    Returns:
    - ndarr: A uint8 numpy array of shape (H, W, 3)
    """
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    return ndarr


def attention_visualizer(img, attn_weights, token):
    """
    Visuailze the attended regions on a single frame from a single query word.
    Inputs:
    - img: Image tensor input, of shape (3, H, W)
    - attn_weights: Attention weight tensor, on the final activation map
    - token: The token string you want to display above the image

    Outputs:
    - img_output: Image tensor output, of shape (3, H+25, W)

    """
    C, H, W = img.shape
    assert C == 3, "We only support image with three color channels!"

    # Reshape attention map
    attn_weights = cv2.resize(
        attn_weights.data.numpy().copy(), (H, W), interpolation=cv2.INTER_NEAREST
    )
    attn_weights = np.repeat(np.expand_dims(attn_weights, axis=2), 3, axis=2)

    # Combine image and attention map
    img_copy = img.float().div(255.0).permute(1, 2, 0).numpy()[:, :, ::-1].copy()
    masked_img = cv2.addWeighted(attn_weights, 0.5, img_copy, 0.5, 0)
    img_copy = np.concatenate((np.zeros((25, W, 3)), masked_img), axis=0)

    # Add text
    cv2.putText(
        img_copy,
        "%s" % (token),
        (10, 15),
        cv2.FONT_HERSHEY_PLAIN,
        1.0,
        (255, 255, 255),
        thickness=1,
    )

    return img_copy


def get_toy_data(path: str = "final_data.json"):
    return json.load(open(path))


def train(
    model,
    train_dataloader,
    val_dataloader,
    loss_func,
    num_epochs,
    batch_size=32,
    warmup_lr=6e-6,
    warmup_interval=1000,
    lr=6e-4,
    device=torch.device("cpu"),
):
    print("Training started...")
    if warmup_interval is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(0.9, 0.995), eps=1e-9
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=warmup_lr, betas=(0.9, 0.995), eps=1e-9
        )
    iteration = 0
    for epoch_num in range(num_epochs):
        epoch_loss = []
        model.train()
        for it in train_dataloader:
            inp, inp_pos, out, out_pos = it
            model = model.to(device)
            inp_pos = inp_pos.to(device)
            out_pos = out_pos.to(device)
            out = out.to(device)
            inp = inp.to(device)
            gnd = out[:, 1:].contiguous().view(-1).long()
            optimizer.zero_grad()

            pred = model(inp.long(), inp_pos, out.long(), out_pos)
            loss = loss_func(pred, gnd)
            epoch_loss.append(loss.item())
            if warmup_interval is not None and iteration == warmup_interval:
                print(
                    f"End of warmup. Swapping learning rates from {warmup_lr} to {lr}"
                )
                for param_group in optimizer.param_groups:
                    warmup_lr = lr
                    param_group["lr"] = lr

            loss.backward()
            optimizer.step()
            iteration = iteration + 1
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        val_loss, val_acc = val(model, val_dataloader, loss_func, batch_size)
        loss_hist = avg_epoch_loss / (batch_size * 4)
        print(
            f"[epoch: {epoch_num+1}]",
            "[loss: ",
            f"{loss_hist:.4f}",
            "]",
            "val_loss: [val_loss ",
            f"{val_loss:.4f}",
            "]",
        )

    return model


def val(model, dataloader, loss_func, batch_size, device=torch.device("cpu")):
    model.eval()
    epoch_loss = []
    num_correct = 0
    total = 0
    for it in dataloader:
        inp, inp_pos, out, out_pos = it

        model = model.to(device)
        inp_pos = inp_pos.to(device)
        out_pos = out_pos.to(device)
        out = out.to(device)
        inp = inp.to(device)
        gnd = out[:, 1:].contiguous().view(-1).long()
        pred = model(inp.long(), inp_pos, out.long(), out_pos)
        loss = loss_func(pred, gnd)

        pred_max = pred.max(1)[1]
        gnd = gnd.contiguous().view(-1)

        n_correct = pred_max.eq(gnd)
        n_correct = n_correct.sum().item()
        num_correct = num_correct + n_correct

        total = total + len(pred_max)
        epoch_loss.append(loss.item())

    avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
    return avg_epoch_loss / (batch_size * 4), n_correct / total


def inference(model, inp_exp, inp_exp_pos, out_pos_exp, out_seq_len):
    model.eval()
    y_init = torch.LongTensor([14]).unsqueeze(0).cuda().view(1, 1)

    ques_emb = model.emb_layer(inp_exp)
    q_emb_inp = ques_emb + inp_exp_pos
    enc_out = model.encoder(q_emb_inp)
    for i in range(out_seq_len - 1):
        ans_emb = model.emb_layer(y_init)
        a_emb_inp = ans_emb + out_pos_exp[:, : y_init.shape[1], :]
        dec_out = model.decoder(a_emb_inp, enc_out, None)
        _, next_word = torch.max(
            dec_out[0, y_init.shape[1] - 1 : y_init.shape[1]], dim=1
        )

        y_init = torch.cat([y_init, next_word.view(1, 1)], dim=1)
    return y_init, model


def draw(data, x, y, ax):
    seaborn.heatmap(
        data,
        xticklabels=x,
        square=True,
        yticklabels=y,
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        ax=ax,
    )
