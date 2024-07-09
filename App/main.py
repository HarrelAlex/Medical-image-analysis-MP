import torch
from torchvision import transforms
from PIL import Image
import google.generativeai as genai
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Generate your key and put instead of 'APIkey'
genai.configure(api_key="APIkey") 

data_transforms = {
    'Training': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'Testing': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

def visualize_model_predictions(model, img_path):
  
    was_training = model.training
    model.eval()

    try:
        img = Image.open(img_path)
    except IOError:
        print(f"Error: Unable to open {img_path}")
        return None

    img = data_transforms['Testing'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
    
    model.train(mode=was_training)
    return f"{class_names[preds[0]]}"

def test(model):

    # Load the new image
    image_path = 'Input/upload/img.jpeg'
    new_image = Image.open(image_path)

    # Preprocess the new image
    new_image = new_image.resize((224, 224))
    new_image = np.array(new_image) / 255.0
    new_image = np.expand_dims(new_image, axis=0)

    rgb_image = np.stack([new_image, new_image, new_image], axis=-1) 
    predictions = model.predict(rgb_image)
    print(predictions[0][0])
    
    predicted_class = "Pneumonia" if predictions[0][0] > 0.6 else "No Pneumonia"
    return predicted_class

def calculate_risk(ans, age, gender, medical_history, lifestyle_factors, blood_pressure, cholesterol, symptoms):
    risk_score = 0
   
    if ans == 'Glioma Tumor':
        risk_score += 50
    elif ans == 'Meningioma Tumor':
        risk_score += 40
    elif ans == 'Pituitary Tumor':
        risk_score += 30
    elif ans == 'No Tumor':
        risk_score += 10
    elif ans == 'Pneumonia':
        risk_score += 50
    elif ans == 'No Pneumonia':
        risk_score += 10

    if age > 60:
        risk_score += 20
    elif age > 40:
        risk_score += 10
 
    if gender == 'Male':
        risk_score += 10
    elif gender == 'Female':
        risk_score += 5
 
    if 'Family History' in medical_history:
        risk_score += 15
    if 'Previous Diagnosis' in medical_history:
        risk_score += 20

    if 'Smoking' in lifestyle_factors:
        risk_score += 10
    if 'Alcohol' in lifestyle_factors:
        risk_score += 5
  
    if blood_pressure > 140:
        risk_score += 10
    if cholesterol > 200:
        risk_score += 10
   
    if 'Severe Headache' in symptoms:
        risk_score += 20
    if 'Vision Problems' in symptoms:
        risk_score += 15
    return risk_score

prompt="""  are a radiologist. You will be analyzing the image provided, using the classification
and risk score provided. Your job is to provide a medical report(make the heading "REPORT: " bold), along with additional resources a 
patient may require for educating themselves about the disease. Also add in capitalized bold letters how soon they 
should seek the consultancy of a medical professional:   """

prompt_2='''You are a chat bot needed to answer queries regarding a disease and the treatment,
You are provided with the medical report, Please answer any following questions based on the disease diagnosed,
the patient's details and the report : '''

def generate_med_report(ans,rf):

    model=genai.GenerativeModel('gemini-pro')
    response = model.generate_content(ans+rf+prompt)
    return response.text

def generate_med_report_qna(med_report, question):

    model=genai.GenerativeModel("gemini-pro")
    response1=model.generate_content(prompt_2+med_report+question)
    return response1.text