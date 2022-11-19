from pprint import pprint
import pickle
import json
import time
from seleniumwire import webdriver  # Import from seleniumwire
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from seleniumwire.utils import decode
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

def process(L):
    for x in L:
        print(x["lat"], x["lng"])


def connection_routine():
    options = Options()
    options.headless = False

    caps = DesiredCapabilities.CHROME
    caps['goog:loggingPrefs'] = {'performance': 'ALL'}


    """driver = webdriver.Chrome(ChromeDriverManager().install(),
                              options=options,
                              desired_capabilities=caps
                              )"""
    driver = webdriver.Chrome(executable_path="chromedriver/chromedriver.exe",
                              options=options,
                              desired_capabilities=caps
                              )

    URL = "https://geoguessr.com/maps/59a1514f17631e74145b6f47"

    driver.get(URL)
    time.sleep(5)

    # Add cookies for login
    cookies = pickle.load(open("cookies.pkl", "rb"))
    for cookie in cookies:
        driver.add_cookie(cookie)

    # need to open it twice to be logged in
    driver.get(URL)
    time.sleep(5)
    driver.refresh()
    return driver


def callback1(driver):
    try:
        element = driver.find_element(By.XPATH, "/html/body/div/div/div[2]/div[1]/main/div/div/div[1]/div[3]/div/div/button").click()
        return 0
    except NoSuchElementException:
        raise EnvironmentError("didnt find 1play button.")
        return 1


def callback2(driver):
    try:
        element = driver.find_element(By.XPATH,"/html/body/div/div/div[2]/div[1]/main/div/div[2]/div/div/div[3]/div/div/button").click()
        return 0
    except NoSuchElementException:
        raise EnvironmentError("didnt find 2play button.")
        return 1



def game_start_routine(driver):
    ret = 1
    while ret == 1:
        time.sleep(1)
        ret = callback1(driver)

    time.sleep(10)
    ret = 1
    while ret == 1:
        time.sleep(1)
        ret = callback2(driver)
    print("Started new game")
    return driver



def network_logs(driver):
    print("-=" * 50)
    # Access requests via the `requests` attribute
    for request in driver.requests:
        if request.response and request.response.body:
            # important to decode
            body = decode(request.response.body, (request.response.headers.get('Content-Encoding', 'identity')))
            try:
                data = json.loads(body)
                L = data["rounds"]
                process(L)
                return L
            except UnicodeDecodeError as u:
                continue
            except Exception as e:
                continue
    return None

if __name__=="__main__":

    driver = connection_routine()
    driver = game_start_routine(driver)
    print("la ferme")
    time.sleep(5)
    while True:
        L = network_logs(driver)
        time.sleep(10)

