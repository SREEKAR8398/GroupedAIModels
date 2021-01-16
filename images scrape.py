import time
from selenium import webdriver

driver = webdriver.chrome("C://Program Files//webdriver//chromedriver.exe")
webpage = "https://thispersondoesnotexist.com/"
driver.get(webpage)

while True:
    time.sleep(5)
    driver.refresh()
