import pickle
import cv2
import time
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch
import torchvision.models as models
from patchify import patchify
import keyboard
import sys
sys.path.append(r"M:\projets_perso\GeoG\dataset_generator")
sys.path.append(r"M:\projets_perso\GeoG\training")
from paths import ROOTDIR, DATADIR
from sample import take_nice_screenshot
from eval import countries_from_output

def connection_routine():
    options = Options()
    options.headless = False
    options.add_argument("--start-maximized")
    options.add_argument('log-level=1')

    caps = DesiredCapabilities.CHROME
    caps['goog:loggingPrefs'] = {'performance': 'ALL'}

    driver = webdriver.Chrome(executable_path="M:/projets_perso/GeoG/dataset_generator/chromedriver/chromedriver.exe",
                              options=options,
                              desired_capabilities=caps
                              )
    URL = "https://geoguessr.com/maps/59a1514f17631e74145b6f47"
    driver.get(URL)
    time.sleep(5)
    # Add cookies for login
    cookies = pickle.load(open("dataset_generator/cookies.pkl", "rb"))
    for cookie in cookies:
        driver.add_cookie(cookie)
    # need to open it twice to be logged in
    driver.get(URL)
    return driver



def predict_ss(driver, model, ohe):
    element = driver.find_element(By.XPATH,
                                  "/html/body/div/div/div/main/div/div/div[1]/div/div/div/div/div[1]/div/div[10]/div/canvas")

    screenshot = element.screenshot_as_png
    with open("pred.png", 'wb') as f:
        f.write(screenshot)
    print("took screenshot")
    input_img = cv2.imread("pred.png")

    patches = patchify(input_img, (456, 456, 3), step=456)
    patches = patches.reshape((-1, 456, 456, 3))
    print("---------")
    preds = []
    for i in range(8):
        patch = patches[i, ...]
        patch = patch[np.newaxis, ...]
        patch = torch.Tensor(patch).to(device)
        patch = torch.permute(patch, (0, 3, 1, 2))
        patch = patch.float()
        pred = model(patch)
        preds.append(pred.cpu().detach().numpy())


        predicted_label = countries_from_output(ohe, pred)


    preds = np.array(preds)
    print("prediction:")
    mean_pred = np.mean(preds, axis=0)

    confidence = float(torch.max(torch.nn.Softmax(dim=1)(torch.Tensor(mean_pred))))
    predicted_country_tag = countries_from_output(ohe, torch.Tensor(mean_pred))

    print(predicted_country_tag)
    print(confidence)

    return



if __name__ == '__main__':
    print("okj")

    driver = connection_routine()



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(f'{ROOTDIR}training/country_tags.pkl', 'rb') as f:
        country_tags = pickle.load(f)
    assert isinstance(country_tags, list)
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(np.array(country_tags).reshape(-1, 1))

    model = models.efficientnet_b5(progress=False)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.1, inplace=True),
        torch.nn.Linear(2048, len(country_tags)),
    )
    model.load_state_dict(
        torch.load(f"{ROOTDIR}training/saved_models/efnetb5/model_epoch_10_loss2.0128148776963024.pth"))
    model.to(device)
    model.eval()

    keyboard.add_hotkey('k', predict_ss, args=[driver, model, ohe])
    while True:
        time.sleep(1)