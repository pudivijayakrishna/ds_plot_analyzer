import google.generativeai as genai

genai.configure(api_key="AIzaSyCPmh6KmUlW3osPSZdvh9uE3bCMjN2UYVg")

print("Checking available models...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"Model Name: {m.name}")