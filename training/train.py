from pprint import pprint
import numpy as np
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
import torch
import torchvision.models as models
from DS_1080p_Country import DS_1080p_Country
from torch.utils.data import Dataset


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

    # base dataset
    ds = DS_1080p_Country("D:/projets_perso/GeoG/datasets/ADW_1080p")

    # train test datasets
    nb_train = int(h['ratio_train_test'] * len(ds))
    print(nb_train)
    train_dataset, test_dataset = torch.utils.data.random_split(ds,
                                                                [nb_train, len(ds) - nb_train],
                                                                torch.Generator(device='cuda'))

    # loaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=h['batch_size'], shuffle=False, num_workers=0,
                                              drop_last=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=h['batch_size'], shuffle=False, num_workers=0,
                                             drop_last=True)

    # model
    torch.hub.set_dir("./cache-dir/")
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)
    model.fc = torch.nn.Linear(512, ds.nb_countries())
    model.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=h['lr'])


    for ep in range(0, h['epochs']):
        print(f'Epoch {ep+1}')

        model.train()

        for i, [patches, target] in enumerate(trainloader):
            image_batch_loss = 0.0
            print(f"training batch ({h['batch_size']} images) number {i}")
            patches.to(device)
            target.to(device)
            target = torch.squeeze(target)
            image_country_prediction = list()
            for j in range(32):
                p = patches[:, j, :, :, :]
                p = torch.Tensor(p)
                p = torch.permute(p, (0, 3, 1, 2))
                p = p.float()

                out = model(p)

                predicted_countries_batch = countries_from_output(ds.ohe, out)
                image_country_prediction.append(predicted_countries_batch)

                optimizer.zero_grad()
                batch_loss = loss(out, target)
                batch_loss.backward()
                optimizer.step()

                image_batch_loss += batch_loss.item()

            print(f"Train Loss of this Image Batch: {image_batch_loss}")
            print("="*100)



        model.eval()
        with torch.no_grad():
            for i, [patches, target] in enumerate(testloader):
                image_batch_loss = 0.0
                print(f"test batch ({h['batch_size']} images) number {i}")
                patches.to(device)
                target.to(device)
                target = torch.squeeze(target)
                image_country_prediction = list()
                for j in range(32):
                    p = patches[:, j, :, :, :]
                    p = torch.Tensor(p)
                    p = torch.permute(p, (0, 3, 1, 2))
                    p = p.float()

                    out = model(p)

                    predicted_countries_batch = countries_from_output(ds.ohe, out)
                    image_country_prediction.append(predicted_countries_batch)

                    optimizer.zero_grad()
                    batch_loss = loss(out, target)
                    batch_loss.backward()
                    optimizer.step()

                    image_batch_loss += batch_loss.item()

                print(f"predictions for this batch of images :")
                pprint(image_country_prediction)
                list_of_winners = []
                image_country_prediction = np.array(image_country_prediction)
                for k in range(h['batch_size']):
                    column = image_country_prediction[:, k].tolist()
                    winner = most_common(column)
                    list_of_winners.append(winner)
                print(f"final verdict of the model for this batch of images:")
                pprint(list_of_winners)
                print(f"ground truth for this batch of images :")
                pprint(countries_from_output(ds.ohe, target))
                print(f"Train Loss of this Image Batch: {image_batch_loss}")
                print("=" * 100)

    return


if __name__ == "__main__":
    hyperparams = {
        'batch_size': 8,
        'epochs': 100,
        'ratio_train_test': 0.99,
        'lr': 0.0001
    }

    train(hyperparams)
