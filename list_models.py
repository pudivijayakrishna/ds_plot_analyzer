import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

print("Checking available models...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"Model Name: {m.name}")
