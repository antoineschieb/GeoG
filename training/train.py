from pprint import pprint
import numpy as np
from torchvision.models import ResNet18_Weights, EfficientNet_B5_Weights
from tqdm import tqdm
import torch
torch.cuda.empty_cache()
import torchvision.models as models
from DS_1080p_Country import PartitionedDataset
from torch.utils.data import Dataset
from paths import ROOTDIR, DATADIR

def countries_from_output(ds_ohe, out):
    m = torch.nn.Softmax(dim=1)
    real_out = m(out)
    L = ds_ohe.inverse_transform(real_out.detach().cpu().numpy())
    return [x[0] for x in L]


def most_common(lst):
    return max(set(lst), key=lst.count)


def train(h):
    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_printoptions(sci_mode=False, precision=2, linewidth=200)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(f"device: {device}")

    # train dataset
    train_ds = PartitionedDataset(
        f"{DATADIR}v3/0", 50, (456, 456, 3), offset=(65, 85), shuffle_buffer=True)
    test_ds = PartitionedDataset(
        f"{DATADIR}ADW_1080p_subset", 1, (456, 456, 3), offset=(65, 85), shuffle_buffer=False, ohe=train_ds.ohe)

    # loaders
    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=h['batch_size'], shuffle=False, num_workers=0,
                                              drop_last=True, generator=torch.Generator(device='cuda'))
    testloader = torch.utils.data.DataLoader(test_ds, batch_size=h['batch_size'], shuffle=False, num_workers=0,
                                             drop_last=True, generator=torch.Generator(device='cuda'))

    # model
    torch.hub.set_dir(f"{ROOTDIR}/cache-dir/")
    model = models.efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1, progress=False)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.1, inplace=True),
        torch.nn.Linear(2048, train_ds.nb_countries()),
    )
    model.to(device)

    loss = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=h['lr'])
    print(f"nb countries in DS: {train_ds.nb_countries()}")
    torch.cuda.memory_summary(device=None, abbreviated=False)
    for ep in range(0, h['epochs']):
        print(f'Epoch {ep+1}')

        model.train()
        epoch_loss = 0.0
        last100BL = 0.0
        for i, [patch, target] in tqdm(enumerate(trainloader), total=len(train_ds)//h['batch_size']):
            target = torch.squeeze(target)
            patch = torch.Tensor(patch)
            patch = torch.permute(patch, (0, 3, 1, 2))
            patch = patch.float()

            optimizer.zero_grad()
            out = model(patch)
            batch_loss = loss(out, target)

            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item()
            last100BL += batch_loss.item()
            if i % 1000 == 0:
                print(f"last 1000 batch loss {last100BL/1000}")
                torch.save(model.state_dict(), f"{ROOTDIR}training/saved_models/latest.pth")
                last100BL = 0.0

        torch.save(model.state_dict(), f"{ROOTDIR}training/saved_models/model_epoch_{ep}_loss{epoch_loss/i}.pth")
        print(f"avg Train Loss for this epoch: {epoch_loss/i}")
        print("="*100)
        print()
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for i, [patch, target] in enumerate(testloader):  # , total=len(test_ds)//h['batch_size']:
                patch.to(device)
                target.to(device)
                target = torch.squeeze(target)

                patch = torch.Tensor(patch)
                patch = torch.permute(patch, (0, 3, 1, 2))
                patch = patch.float()

                out = model(patch)

                batch_loss = loss(out, target)
                test_loss += batch_loss.item()
        print(f"avg Test Loss for this epoch: {test_loss/i}")
        print("=" * 100)
        print()


    return


if __name__ == "__main__":
    hyperparams = {
        'batch_size': 2,
        'epochs': 300,
        'lr': 0.0001
    }

    train(hyperparams)
