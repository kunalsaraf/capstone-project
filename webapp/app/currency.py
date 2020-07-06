import matplotlib.pyplot as plt
# import uuid
# import boto3
# import pickle
# import datetime
# from PIL import Image
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
global model
MODEL_PATH = './app/model/model.h5'
model = load_model(MODEL_PATH)
    
def get_label(file_path):
    processed_image = process_image(file_path)
    classes = model.predict(processed_image)
#    print(classes)
    predicted_labels = classes.tolist()[0]
    denomination=[200, 2000, 500]
    percentage_labels=[0.0 ,0.0 ,0.0]
    sum_total=sum(predicted_labels)
    for i in range(0,len(predicted_labels)):
        percentage_labels[i]=round((predicted_labels[i]/sum_total)*100,2)
    result_dict = dict(zip(denomination,percentage_labels))
    print(result_dict)
    denomination.clear()
    percentage_labels.clear()
    for i in sorted(result_dict):
        denomination.append(str(i))
        percentage_labels.append(result_dict[i])
        print((i, result_dict[i]), end =" ") 
    plt.bar(denomination, percentage_labels, color = ['red', 'green', 'yellow'])
    plt.title('Probability of each denomination')
    plt.xlabel('Denominations')
    plt.ylabel('Percentage (%)')
    plt.grid(b=True)
    plt.savefig("./app/static/result/currency.jpg")
    plt.close()
    # plt.show()

def process_image(file_path):
    img = load_img(file_path, target_size=(128, 96, 3))
    img = img_to_array(img)
    img = img.reshape(1,128,96,3).astype('float')
    img /= 255
    return img

# def upload_file_to_s3(file_path, filename):
#     compressed = compress_image(file_path)
#     if compressed:
#         try:
#             s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
#             with open(file_path, 'rb') as image:
#                 s3.upload_fileobj(image, BUCKET, UPLOAD_DIR + filename)
#                 return True
#         except:
#             return False
#     else:
#             return False
#
# def save_prediction_to_s3(image_filename, predictions):
#     filename = str(uuid.uuid4())
#     timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#     data = [timestamp, UPLOAD_DIR + image_filename]
#     data += predictions
#     data = pickle.dumps(data)
#     try:
#         s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
#         s3.put_object(Body=data, Bucket=BUCKET, Key=PRED_DIR + filename)
#         return True
#     except:
#         return False
#
# def compress_image(file_path):
#     try:
#         img = Image.open(file_path)
#         img_width = img.size[0] * 20 // 100
#         img_height = img.size[1] * 20 // 100
#         img = img.resize((img_width,img_height),Image.ANTIALIAS)
#         img.save(file_path, optimize=True,quality=50)
#         return True
#     except FileNotFoundError:
#         return False
#
# def allowed_file(filename):
# 	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
if __name__=="__main__":
    get_label('./model/demo3.jpg')
