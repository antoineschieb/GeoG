from random import randrange
import sys
import os
import pickle
import json
import time
from seleniumwire import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from seleniumwire.utils import decode
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service

def process(list_of_Ls):
    unique_locs = []
    for L in list_of_Ls:
        for x in L:
            e = [x["lat"], x["lng"], x["streakLocationCode"]]
            if e not in unique_locs:
                unique_locs.append(e)
    return unique_locs


def connection_routine():
    options = Options()
    options.headless = True
    options.add_argument("--start-maximized")
    options.add_argument('log-level=1')

    caps = DesiredCapabilities.CHROME
    caps['goog:loggingPrefs'] = {'performance': 'ALL'}

    driver = webdriver.Chrome(executable_path="dataset_generator/chromedriver/chromedriver.exe",
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

def click_button(driver, xpath):
    seconds = 0
    while seconds < 120:
        try:
            element = driver.find_element(By.XPATH, xpath)
            element.click()
            return
        except NoSuchElementException:
            time.sleep(1)
            seconds += 1
    r = randrange(0, 10)
    driver.get_screenshot_as_file(f"crashhh{r}.png")
    raise TimeoutError(f"Timeout for button {xpath}")






def game_start_routine(driver, change_settings=False):
    # click_button(driver, "/html/body/div/div/div[2]/div[1]/main/div/div/div[1]/div[3]/div/div/button")
    driver.get("https://www.geoguessr.com/maps/59a1514f17631e74145b6f47/play")
    if change_settings:
        click_button(driver, "/html/body/div/div/div[2]/div[1]/main/div/div[2]/div/div/div[5]/div/div/div[2]/input")
        click_button(driver, "/html/body/div/div/div[2]/div[1]/main/div/div[2]/div/div/div[5]/div/div[2]/div/div[2]/label[1]/div[3]/input")
    click_button(driver, "/html/body/div/div/div[2]/div[1]/main/div/div[2]/div/div/div[3]/div/div/button")
    return driver


def hide_class_name(driver, cname):
    driver.execute_script(
        f'document.getElementsByClassName("{cname}")[0].style.visibility = "hidden";')
    return


def take_nice_screenshot(driver, name):
    # driver.get_screenshot_as_file(f"datasets/v1/{name}.png")
    # element = driver.find_element_by_css("
    # html body div#__next div.version3-in-game_root__P2ydF.version3-in-game_backgroundDefault__oNQrE div.version3-in-game_content__9t8Xc main.version3-in-game_layout__Hi_Iw div.game-layout div.game-layout__canvas div.game-layout__panorama div.game-layout__panorama-canvas div div div.gm-style div div div div.mapsConsumerUiSceneInternalCoreScene__root.widget-scene canvas.mapsConsumerUiSceneInternalCoreScene__canvas.widget-scene-canvas
    # ")


    element = driver.find_element(By.XPATH, "/html/body/div/div/div/main/div/div/div[1]/div/div/div/div/div[1]/div/div[10]/div/canvas")
    # hide elements
    """
    to_hide = ["status_inner__1eytg",
               "panorama-compass_compassContainer__MEnh0",
               "guess-map__canvas-container",
               "guess-map__guess-button",
               "tooltip_reference__qDBCi",
               "styles_control__zEkd0",
               ]

    for h in to_hide:
        hide_class_name(driver, h)

    time.sleep(20)"""
    screenshot = element.screenshot_as_png
    with open(f"D:/projets_perso/GeoG/datasets/v3/{name}.png", 'wb') as f:
        f.write(screenshot)
    return


def network_logs(driver):
    list_of_Ls = []
    # Access requests via the `requests` attribute
    for request in driver.requests:
        if request.response and request.response.body:
            # important to decode
            body = decode(request.response.body, (request.response.headers.get('Content-Encoding', 'identity')))

            try:
                data = json.loads(body)
                L = data["rounds"]
                list_of_Ls.append(L)
            except UnicodeDecodeError as u:
                continue
            except Exception as e:
                continue
    return list_of_Ls



def task():
    driver = connection_routine()
    driver = game_start_routine(driver, change_settings=True)
    # driver.set_window_size(1920, 1080)

    for game in range(10):
        driver.set_window_size(1920, 1080)
        #driver.maximize_window()
        time.sleep(10)
        for r in range(1, 6):
            time.sleep(7)
            while True:
                list_of_Ls = network_logs(driver)
                unique_locs = process(list_of_Ls)
                if len(unique_locs) != 0:
                    if len(unique_locs) == 5*game + r:
                        break
            time.sleep(2)
            ssname = f"{unique_locs[-1][0]}_{unique_locs[-1][1]}_{unique_locs[-1][2]}"
            take_nice_screenshot(driver, ssname)
            print(f"{os.getpid()}:{ssname}")
            # click map
            click_button(driver, "/html/body/div/div/div/main/div/div/div[4]/div/div[3]/div/div/div/div/div[2]/div[2]")
            # guess button
            click_button(driver, "/html/body/div/div/div/main/div/div/div[4]/div/div[4]/button/div")
            if r < 5:
                # next round
                click_button(driver, "/html/body/div/div/div/main/div[2]/div/div[2]/div/div[1]/div/div[4]/button/div")
        time.sleep(30) # avoid "slow down there cowboy"
        game_start_routine(driver, change_settings=False)

    return "10 games done"


if __name__=='__main__':
    options = Options()
    options.headless = False
    options.add_argument("--start-maximized")
    options.add_argument('log-level=1')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--disable-blink-features=AutomationControlled')
    print(1)
    caps = DesiredCapabilities.CHROME
    caps['goog:loggingPrefs'] = {'performance': 'ALL'}
    s = Service(r"M:\projets_perso\GeoG\dataset_generator\chromedriver\chromedriver.exe")


    driver = webdriver.Chrome(executable_path=r"M:\projets_perso\GeoG\dataset_generator\chromedriver\chromedriver.exe",
                              options=options,
                              desired_capabilities=caps,
                              service=s
                              )
    print(3)
    URL = "https://geoguessr.com/maps/59a1514f17631e74145b6f47"
    print(4)
    driver.get(URL)
    print(5)

