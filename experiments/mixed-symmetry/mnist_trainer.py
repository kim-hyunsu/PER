import torch
import numpy as np
import argparse
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
from emlp.groups import Z, DirectProduct, Group
from emlp.reps.linear_operators import Rot90
from emlp.reps import T, Vector, Scalar
import objax
import objax.functional as F
from tqdm import tqdm
import jax.numpy as jnp
import wandb
from rpp.classifiers import MLPClassifier, SoftEMLPClassifier, SoftEMLPBlock


class Zr(Group):
    def __init__(self, n):
        rotation = Rot90(n, 1)
        self.discrete_generators = [rotation]
        super().__init__(n)


def main(args):

    lr = args.lr
    seed = 2022 + args.trial
    torch.manual_seed(seed)
    np.random.seed(seed)

    watermark = "{}_{}_{}_{}_{}".format(
        args.network, args.rglr, args.equiv, args.wd, args.trial)
    wandb.init(
        project="Mixed Symmetry, MNIST",
        name=watermark,
        mode="disabled" if args.logoff else "online",
    )
    wandb.config.update(args)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda img: torch.nn.functional.avg_pool2d(img, 2, 2),
        lambda img: torch.nn.functional.avg_pool2d(img, 5, 1),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda img: torch.nn.functional.avg_pool2d(img, 2, 2),
        lambda img: torch.nn.functional.avg_pool2d(img, 5, 1),
    ])

    dataset = torchvision.datasets.MNIST(args.data_path,
                                         train=True, download=True,
                                         transform=transform_train)
    train_set, valid_set = torch.utils.data.random_split(dataset, [
                                                         50000, 10000])
    trainloader = DataLoader(train_set, shuffle=True,
                             batch_size=args.batch_size)
    validloader = DataLoader(valid_set, shuffle=False,
                             batch_size=args.batch_size)

    testset = torchvision.datasets.MNIST(args.data_path,
                                         train=False, download=False,
                                         transform=transform_test)
    testloader = DataLoader(testset, shuffle=False, batch_size=args.batch_size)

    h, w = 10, 10
    classes = 10
    G = DirectProduct(Z(h), Z(w))
    G1 = Zr(h)
    if args.network.lower() == "mlp":
        print("Using MLP", flush=True)
        model = MLPClassifier(h, w, classes, F.relu)
    elif args.network.lower() == "softemlp":
        print("Using SoftEMLP", flush=True)
        model = SoftEMLPClassifier(
            (h, w), Vector, Vector, classes*Scalar, F.relu, (G, G1))
    else:
        raise Exception("Invalid Network")

    ## training setup ##
    opt = objax.optimizer.Adam(model.vars())

    @objax.Jit
    @objax.Function.with_vars(model.vars())
    def CrossEntropyLoss(x, y):
        yhat = model(x)
        loss = F.loss.cross_entropy_logits_sparse(yhat, y)
        pred = yhat.argmax(1)
        acc = (pred == y).mean()
        return loss.mean(), acc

    @objax.Jit
    @objax.Function.with_vars(model.vars())
    def loss_fn(x, y):
        yhat = model(x)
        rglr = 0
        if args.network.lower() in ["softemlp"]:
            for i, lyr in enumerate(model.network):
                if not isinstance(lyr, SoftEMLPBlock):
                    Pw = lyr.Pw
                    Pw1 = lyr.Pw1
                    Pb = lyr.Pb
                    Pb1 = lyr.Pb1
                else:
                    Pw = model.Pw
                    Pw1 = model.Pw1
                    Pb = model.Pb
                    Pb1 = model.Pb1
                W1 = lyr.w.value.reshape(-1)
                W2 = Pw1@Pw@lyr.w.value.reshape(-1)
                Wdiff = W1-W2
                b1 = lyr.b.value
                b2 = Pb1@Pb@lyr.b.value
                bdiff = b1-b2
                rglr += 0.5*(Wdiff*(Pw@Wdiff)).sum() + \
                    0.5*(bdiff*(Pb@bdiff)).sum()
        elif args.network.lower() in ["rpp"]:
            pass
        loss = F.loss.cross_entropy_logits_sparse(yhat, y)
        rglr = args.rglr*rglr
        return loss.mean()+rglr

    grad_and_val = objax.GradValues(loss_fn, model.vars())

    @objax.Jit
    @objax.Function.with_vars(model.vars()+opt.vars())
    def train_op(x, y, lr):
        g, v = grad_and_val(x, y)
        opt(lr=lr, grads=g)
        return v

    top_acc = float("inf")
    top_test_acc = float("inf")
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        time_ep = time.time()
        train_loss = 0
        for x, y in trainloader:
            l = train_op(jnp.array(x), jnp.array(y), lr)
            train_loss += l[0]*x.shape[0]
        train_loss /= len(trainloader.dataset)
        valid_loss = 0
        valid_acc = 0
        for x, y in validloader:
            l, acc = CrossEntropyLoss(jnp.array(x), jnp.array(y))
            valid_loss += l*x.shape[0]
            valid_acc += acc*x.shape[0]
        valid_loss /= len(validloader.dataset)
        valid_acc /= len(validloader.dataset)

        time_ep = time.time() - time_ep

        if valid_acc < top_acc:
            top_acc = valid_acc
            test_acc = 0
            for x, y in testloader:
                _, acc = CrossEntropyLoss(jnp.array(x), jnp.array(y))
                test_acc += acc*x.shape[0]
            test_acc /= len(testloader.dataset)
            top_test_acc = test_acc

        wandb.log({"train_loss": train_loss,
                  "valid_loss": valid_loss, "valid_acc": valid_acc,  "time": time_ep})

    print("Top Test Acc:", top_test_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Runner")
    parser.add_argument(
        "--network",
        type=str,
        default="SoftEMLP",
        help="rpp, softemlp, mlp",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="just a flag to distinguish models",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="training epochs",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./datasets/",
        help="",
    )
    parser.add_argument(
        "--basic_wd",
        type=float,
        default=1e-2,
        help="basic weight decay",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
    )
    parser.add_argument(
        "--equiv",
        type=float,
        default=200,
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--logoff",
        action="store_true"
    )
    parser.add_argument(
        "--rglr",
        type=float,
        default=1
    )

    args = parser.parse_args()
    main(args)
