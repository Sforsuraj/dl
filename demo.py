import requests

url = "https://raw.githubusercontent.com/Sforsuraj/dl/main/README.md"
code = requests.get(url).text

print(code)
