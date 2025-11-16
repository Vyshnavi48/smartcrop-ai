import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(torch.load('model.pt', map_location='cpu'))
    model.eval()
    return model

model = load_model()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['Bacterial Spot', 'Early Blight', 'Healthy', 'Late Blight', 'Leaf Mold',
           'Septoria Spot', 'Spider Mites', 'Target Spot', 'Mosaic Virus', 'Yellow Leaf Curl']

st.title("SmartCrop AI")
st.write("Upload a tomato leaf image")

uploaded = st.file_uploader("Choose image...", type=['jpg', 'jpeg', 'png'])
if uploaded:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption="Uploaded Leaf", use_column_width=True)
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(x)
        pred = output.argmax(1).item()
        prob = torch.softmax(output, 1)[0][pred].item()
    st.write(f"**Prediction:** {classes[pred]}")
    st.write(f"**Confidence:** {prob:.1%}")
    if classes[pred] == "Healthy":
        st.success("Healthy Plant!")
    else:
        st.error("Disease Detected!")
        
