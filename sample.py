from pprint import pprint
import pickle
import json
import time
from seleniumwire import webdriver  # Import from seleniumwire
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from seleniumwire.utils import decode

def process(L):
    print("==========="*20)
    for x in L:
        print(x["lat"], x["lng"])



if __name__=="__main__":

    caps = DesiredCapabilities.CHROME
    caps['goog:loggingPrefs'] = {'performance': 'ALL'}

    # Create a new instance of the Chrome driver
    driver = webdriver.Chrome(executable_path="chromedriver/chromedriver.exe",
                              desired_capabilities=caps,
                              )

    # Go to the Google home page
    driver.get('https://www.geoguessr.com/game/g4RZVZAtJvBZd93V')

    # Add cookies for login
    cookies = pickle.load(open("cookies.pkl", "rb"))
    for cookie in cookies:
        driver.add_cookie(cookie)

    time.sleep(9)
    driver.get('https://www.geoguessr.com/game/g4RZVZAtJvBZd93V')
    time.sleep(9)
    print("g")




    while True:
        print("-="*50)
        # Access requests via the `requests` attribute
        for request in driver.requests:
            if request.response and request.response.body:
                # important to decode
                body = decode(request.response.body, (request.response.headers.get('Content-Encoding', 'identity')))
                try:
                    data = json.loads(body)
                    L = data["rounds"]
                    process(L)
                except UnicodeDecodeError as u:
                    pass
                except Exception as e:
                    pass

        time.sleep(10)
