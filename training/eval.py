import numpy as np
import cv2
import torch
import torchvision.models as models
from DS_1080p_Country import DS_1080p_Country, PartitionedDataset
from torchvision.models import ResNet18_Weights



def countries_from_output(ohe, out):
    m = torch.nn.Softmax(dim=1)
    real_out = m(out)
    L = ohe.inverse_transform(real_out.detach().cpu().numpy())
    return [x[0] for x in L]


def evaluate(N, ohe, small_ds):
    [orig_patch, country_label] = small_ds[N]

    patch = orig_patch[np.newaxis, ...]
    patch = torch.Tensor(patch)

    patch = torch.permute(patch, (0, 3, 1, 2))
    patch = patch.float()

    pred = model(patch)
    confidence = float(torch.max(torch.nn.Softmax(dim=1)(pred)))
    country_label = countries_from_output(ohe, country_label)
    predicted_label = countries_from_output(ohe, pred)
    cv2.imwrite(f"training/full_img_eval/p{N}_{country_label[0]}_{predicted_label[0]}_{round(confidence, 2)}.png", orig_patch)
    return pred


if __name__ == "__main__":

    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_printoptions(sci_mode=False, precision=2, linewidth=200)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(f"device: {device}")

    # base dataset
    ds = PartitionedDataset("D:/projets_perso/GeoG/datasets/v3", 1)
    small_ds = PartitionedDataset("D:/projets_perso/GeoG/datasets/v3_eval", 1, ohe=ds.ohe)
    # model
    torch.hub.set_dir("./cache-dir/")
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT, progress=False)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(512, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.Linear(256, ds.nb_countries())
    )
    model.load_state_dict(torch.load("M:/projets_perso/GeoG/saved_models/model_epoch_6_loss110260.72841894627.pth"))
    model.to(device)
    model.eval()

    with torch.no_grad():
        for imgidx in range(20, 50):
            M = 32*imgidx
            preds = []
            print(f'img index: {imgidx}')
            for i in range(32):
                pred = evaluate(M+i, ds.ohe, small_ds)
                preds.append(pred.cpu().detach().numpy())
            preds = np.array(preds)
            print("prediction:")
            mean_pred = np.mean(preds, axis=0)
            print(countries_from_output(ds.ohe, torch.Tensor(mean_pred)))

