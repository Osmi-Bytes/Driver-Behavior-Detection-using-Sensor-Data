import requests
import json

url = "http://localhost:11434/api/generate"

payload = json.dumps({
  "model": "qwen3:0.6b",
  "prompt": "Hi I am Osama"
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
