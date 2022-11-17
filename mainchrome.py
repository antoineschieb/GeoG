# from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

from seleniumwire import webdriver

import time
import pickle
import json
from pprint import pprint


def connection_routine():
    options = Options()
    options.headless = False

    caps = DesiredCapabilities.CHROME
    caps['goog:loggingPrefs'] = {'performance': 'ALL'}


    driver = webdriver.Chrome(ChromeDriverManager().install(),
                              options=options,
                              desired_capabilities=caps)

    URL = "https://www.geoguessr.com/maps/59a1514f17631e74145b6f47"

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


def process_browser_log_entry(entry):
    response = json.loads(entry['message'])['message']
    return response


def network_logs(driver):
    logs_raw = driver.get_log("performance")
    logs = [json.loads(lr["message"])["message"] for lr in logs_raw]

    def log_filter(log_):
        return (
            # is an actual response
                log_["method"] == "Network.responseReceived"
                # and json
                and "json" in log_["params"]["response"]["mimeType"]
        )
    for log in filter(log_filter, logs):
        request_id = log["params"]["requestId"]
        resp_url = log["params"]["response"]["url"]
        print(f"Caught {resp_url}")
        # pprint(log["params"]["type"])
        # print("=-"*20)
        print(driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": request_id}))


if __name__ == "__main__":
    d = connection_routine()
    d = game_start_routine(d)
    while True:
        print(network_logs(d))
        time.sleep(15)
