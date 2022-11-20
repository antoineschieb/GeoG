from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.firefox.options import Options
import time
import pickle

# Open the website
# driver = webdriver.Firefox(executable_path="M:/projets_perso/TA/webdriver/geckodriver")
options = Options()
options.headless = False
driver = webdriver.Firefox(options=options, service=Service(GeckoDriverManager().install()))


# vraie game
URL = "https://geoguessr.com/maps/59a1514f17631e74145b6f47"


# open bga
driver.get(URL)

while True:
    time.sleep(6)
    print("trying to write cookies...")
    try:
        pickle.dump(driver.get_cookies(), open("cookies.pkl", "wb"))
    except Exception as e:
        raise e ("xd")
