# from twilio.rest import Client
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

print("""
1: to send whatsapp message
2: to send text message
3: to send email
4: to create a table
5: to send bulk whatsapp message
6: convert text to audio    
7: capture the image
8: rainbow 
9: html form
10: search_engine
11: Create filter images
12: Sunglass and web
13: Crop your pic in corner
14: Custom image from numpy
15: Convert english to french
16: Blur the image
17: Volume control
18: Get location and coordinates
19: Bulk Email
20: top 5 search on google
21: Click picture using Javascript
22: Search engine using Javsscript
23: Access mic using Javascript
24: Live stream using Javascript
25: Audio to text using Javascript
 """)


ch=input("Enter your choice: ")

def whatsapp():
        import pywhatkit
        pywhatkit.sendwhatmsg_instantly("+918619280612","Hello bro")
        
def text():
            from twilio.rest import Client

            # Your Twilio account SID, Auth Token, and Twilio phone number
            account_sid = 'AC5d3a29dae4fcd2d4db140106db6ce18c'
            auth_token = 'fb3922a52cbd213f97b29453b80c8a1e'
            twilio_phone_number = '+13343104412'

            # Data
            names = ["anushka", "aditya", "aryan", "yash"]
            cities = ["jaigarh", "ajmer", "jaipur", "jaipur"]
            phone_numbers = ["7976971372", "8619280612", "9058451223","7737800033"]

            # Message to be sent
            message = "Welcome to Pink City, Jaipur"

            # Initialize Twilio client
            client = Client(account_sid, auth_token)

            # Function to send SMS message
            def send_sms_message(phone_number, message):
                try:
                    client.messages.create(
                        body=message,
                        from_=twilio_phone_number,
                        to=f"+91{phone_number}"
                        
                    )
                    print(f"Message sent to +91{phone_number}")
                except Exception as e:
                    print(f"Failed to send message to +91{phone_number}: {e}")

            # Loop through all entries and send message to non-Jaipur residents
            for name, city, number in zip(names, cities, phone_numbers):
                if city.lower() != "jaipur":
                    send_sms_message(number, message)
                    print(f"Message sent to {name} in {city}")
                    
def emails():
        import smtplib

        email = input("Sender email :")
        receiver_email = input("Rec. email: ")

        subject=input("Sub : ")
        message=input("Msg : ")

        text = f'Subject: {subject}\n\n {message}'

        server= smtplib.SMTP("smtp.gmail.com",587)
        server.starttls()

        server.login(email,"") #use app password
        server.sendmail(email,receiver_email,text)

        print("Email has been sent to "+receiver_email)

def table():
        from prettytable import PrettyTable
        table = PrettyTable()
        table.add_column("Name", ["anushka","aditya","aryan","yash","shuti","sparsh","jatin","nikhil","anushtha","ankit","sanjeev","rahul","anushka","priyanka","kunal","neeraj"])
        table.add_column("City", ["jaipur","jaipur","jaipur","jaipur","Bhopal","delhi","agra","jaipur","chandigarh","ajmer","jaipur","jaipur","pilani","jaipur","mumbai","jaipur"])
        table.add_column("College", ["skit","CU","CU","CU","VGU","JECRC","VGU","GIT","GIT","JECRC","SKIT","MUJ","BITS","MUJ","SKIT","ST.WILFRID"])
        table.add_column("Phone Number", ["7976971372","8619280612","9058451223","7737800033","7023788003","6350382356","9821972494","9350649498","9821076429","9930807348","7627095917","9359733858","8233120900","6378827581","78827581","7841832374"])

        print(table)
        
def bulkwhats():
        import pywhatkit
        import time

        phone_numbers = ["+917976971372", "+9186900 37470", "+91 77378 00033"]  # Add your phone numbers here
        message = "Hello, welcome to linux world"

        for number in phone_numbers:
            try:
                pywhatkit.sendwhatmsg_instantly(number, message)
                print(f"Message sent to {number}")
                time.sleep(6)  # Sleep for 6 seconds to avoid issues with WhatsApp Web
            except Exception as e:
                print(f"Failed to send message to {number}: {e}")

def speech():

        # Import the required module for text 
        # to speech conversion
        from gtts import gTTS

        # This module is imported so that we can 
        # play the converted audio
        import os

        # The text that you want to convert to audio
        mytext = 'Aur bta bhai kya haal hai?'

        # Language in which you want to convert
        language = 'en'

        # Passing the text and language to the engine, 
        # here we have marked slow=False. Which tells 
        # the module that the converted audio should 
        # have a high speed0
        myobj = gTTS(text=mytext, lang=language, slow=False)

        # Saving the converted audio in a mp3 file named
        # welcome 
        myobj.save("welcome.mp3")

        # Playing the converted file
        os.system("start welcome.mp3")

def images():
        import cv2
        cap=cv2.VideoCapture(0)
        status,pic= cap.read()
        cv2.imshow("Aditya PIC",pic)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        cap.release()

def rainbow():
        import sys
        import itertools

        def print_rainbow_text(text):
            colors = ['\033[91m', '\033[93m', '\033[92m', '\033[96m', '\033[94m', '\033[95m']
            reset_color = '\033[0m'
            
            color_cycle = itertools.cycle(colors)
            
            rainbow_text = ''.join(next(color_cycle) + char for char in text)
            
            sys.stdout.write(rainbow_text + reset_color + '\n')
            sys.stdout.flush()

        # Example usage
        print_rainbow_text("Aditya Singh")

def htmlcode():
        import os
        os.system("chrome file:///C:/Users/Aadi/Desktop/HTML_Daga/first.html" )

def search():
        import os
        os.system("chrome file:///C:/Users/Aadi/Desktop/HTML_Daga/Adityasearch.html" )
    

def filter():
    from PIL import Image, ImageFilter, ImageEnhance

    # Open an image file
    image_path = r"C:\Users\Aadi\Desktop\anushka.jpg"
    image = Image.open(image_path)

    # Apply different filters

    # 1. Blur filter
    blurred_image = image.filter(ImageFilter.BLUR)
    blurred_image.show()  # Display the blurred image

    # 2. Contour filter
    contour_image = image.filter(ImageFilter.CONTOUR)
    contour_image.show()  # Display the contour image

    # 3. Detail filter
    detail_image = image.filter(ImageFilter.DETAIL)
    detail_image.show()  # Display the detailed image

    # 4. Sharpen filter
    sharpened_image = image.filter(ImageFilter.SHARPEN)
    sharpened_image.show()  # Display the sharpened image

    # 5. Edge Enhance filter
    edge_enhanced_image = image.filter(ImageFilter.EDGE_ENHANCE)
    edge_enhanced_image.show()  # Display the edge enhanced image

    # 6. Emboss filter
    embossed_image = image.filter(ImageFilter.EMBOSS)
    embossed_image.show()  # Display the embossed image

    # 7. Brightness enhancement
    enhancer = ImageEnhance.Brightness(image)
    bright_image = enhancer.enhance(1.5)  # Increase brightness by 50%
    bright_image.show()  # Display the brighter image

    # 8. Contrast enhancement
    enhancer = ImageEnhance.Contrast(image)
    contrast_image = enhancer.enhance(2.0)  # Double the contrast
    contrast_image.show()  # Display the high contrast image

    # 9. Color enhancement
    enhancer = ImageEnhance.Color(image)
    color_image = enhancer.enhance(1.5)  # Increase color intensity by 50%
    color_image.show()  # Display the color enhanced image

    # 10. Grayscale filter
    gray_image = image.convert("L")
    gray_image.show()  # Display the grayscale image

    # Save the filtered images if needed
    blurred_image.save("blurred_image.jpg")
    contour_image.save("contour_image.jpg")
    detail_image.save("detail_image.jpg")
    sharpened_image.save("sharpened_image.jpg")
    edge_enhanced_image.save("edge_enhanced_image.jpg")
    embossed_image.save("embossed_image.jpg")
    bright_image.save("bright_image.jpg")
    contrast_image.save("contrast_image.jpg")
    color_image.save("color_image.jpg")
    gray_image.save("gray_image.jpg")

def sunglass():
    from PIL import Image

    # Open the main image
    image_path = r"C:\Users\Aadi\Desktop\anushka.jpg" 
    image = Image.open(image_path)

    # Open the accessory images
    sunglasses_path = r"C:\Users\Aadi\Desktop\sunglass.jpg"
    cap_path = r"C:\Users\Aadi\Desktop\cap.jpg"

    sunglasses = Image.open(sunglasses_path).convert("RGBA")
    cap = Image.open(cap_path).convert("RGBA")

    # Resize accessory images if necessary
    sunglasses = sunglasses.resize((300, 150))  # Adjust the size to fit the face
    cap = cap.resize((300, 150))  # Adjust the size to fit the head

    # Create a mask from the alpha channel of the accessory images
    sunglasses_mask = sunglasses.split()[3]
    cap_mask = cap.split()[3]

    # Paste the sunglasses onto the main image at the desired position
    sunglasses_position = (500, 240)  # Adjust these values based on your image
    image.paste(sunglasses, sunglasses_position, sunglasses_mask)

    # Paste the cap onto the main image at the desired position
    cap_position = (180, 50)  # Adjust these values based on your image
    image.paste(cap, cap_position, cap_mask)

    # Show the final image
    image.show()

    # Save the final image
    image.save("image_with_accessories.jpg")

def cornercrop():
    from PIL import Image, ImageDraw
    import cv2
    import numpy as np

    # Load the image
    image_path =  r"C:\Users\Aadi\Desktop\anushka.jpg"
    image = Image.open(image_path)

    # Convert the image to a format compatible with OpenCV
    opencv_image = np.array(image)

    # Convert RGB to BGR (since OpenCV uses BGR format)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    # Load the pre-trained face detection model (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if at least one face is detected
    if len(faces) > 0:
        # Get the first face detected (x, y, width, height)
        (x, y, w, h) = faces[0]
        
        # Crop the face from the image
        face = image.crop((x, y, x + w, y + h))
        
        # Define the position to paste the face in the upper-left corner
        position = (0, 0)
        
        # Create a blank image the same size as the original for drawing
        blank_image = Image.new("RGBA", image.size)
        
        # Paste the face on the blank image at the specified position
        blank_image.paste(face, position)
        
        # Composite the original image with the face image
        combined_image = Image.alpha_composite(image.convert("RGBA"), blank_image)

        # Convert back to RGB (if needed)
        combined_image = combined_image.convert("RGB")

        # Show the final image with the face pasted in the upper corner
        combined_image.show()

        # Save the final image
        combined_image.save("image_with_face_in_corner.jpg")
    else:
        print("No face detected in the image.")

def numpyimg():
    import numpy as np
    from PIL import Image

    # Define the dimensions of the image
    width, height = 300, 300

    # Create a 3D NumPy array of shape (height, width, 3) for an RGB image
    # Initialize the array with zeros (black color)
    image_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Example: Create a gradient image (vertical gradient from black to white)
    for y in range(height):
        for x in range(width):
            # Set each pixel's color to (R, G, B)
            image_array[y, x] = [x % 256, y % 256, (x + y) % 256]

    # Convert the NumPy array to an Image object
    image = Image.fromarray(image_array)

    # Show the image
    image.show()

    # Save the image to a file
    image.save("custom_image.png")

def engtofrench():
    import requests

    API_URL = "https://api-inference.huggingface.co/models/t5-base"
    API_TOKEN = "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"  # Replace with your actual API token

    headers = {"Authorization": f"Bearer {API_TOKEN}"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    text="my name is Aditya"


    prompt=f"translate English to French: {text}"
    data = query({"inputs": prompt})
    print(data)

def faceblur():
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    def show_frame(frame):
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    # Load the pre-trained face detection Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) for the face
            face_roi = frame[y:y+h, x:x+w]

            # Apply Gaussian blur to the face ROI
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 20)

            # Place the blurred face back into the original frame
            frame[y:y+h, x:x+w] = blurred_face

        # Display the result using matplotlib
        show_frame(frame)

        # Exit the loop when the ENTER key is pressed
        if cv2.waitKey(1) == 13:
            break

# Release the video capture
    cap.release()
    cv2.destroyAllWindows()
    
def VolControl():
    from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import IAudioEndpointVolume

    # Get the default audio device
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)

    volume = cast(interface, POINTER(IAudioEndpointVolume))

    def set_volume(level):
        """
        Set system volume to a specific level
        :param level: float between 0.0 (min) and 1.0 (max)
        """
        volume.SetMasterVolumeLevelScalar(level, None)

    # def get_current_volume():
    #     """
    #     Get current system volume
    #     :return: float between 0.0 (min) and 1.0 (max)
    #     """
    #     return volume.GetMasterVolumeLevelScalar()





    if __name__ == "__main__":
        # Example Usage:
        
        setvol= int(input("Enter the volume : "))
        x = setvol/100
        set_volume(x)
        print(f"Volume set to {setvol}%")
        
def location():
    import geocoder

    def get_current_location():
        # Get the current location based on the public IP address
        g = geocoder.ip('me')

        if g.ok:
            location = g.city + ", " + g.country
            latitude = g.latlng[0]
            longitude = g.latlng[1]
            
            return {
                "location": location,
                "latitude": latitude,
                "longitude": longitude
            }
        else:
            return {
                "error": "Unable to get location data."
            }

    # Example usage
    current_location = get_current_location()
    print(current_location)
    
def bulkEmail():
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    def send_bulk_email(sender_email, app_password, recipients, subject, message_body):
        # Set up the server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        
        # Login to the email server
        try:
            server.login(sender_email, app_password)
        except smtplib.SMTPAuthenticationError as e:
            print("Failed to login:", e)
            return
        
        # Create the message template
        message = MIMEMultipart()
        message['From'] = sender_email
        message['Subject'] = subject
        message.attach(MIMEText(message_body, 'plain'))

        # Send emails to all recipients
        for recipient in recipients:
            try:
                message['To'] = recipient
                server.sendmail(sender_email, recipient, message.as_string())
                print(f"Email sent to {recipient}")
            except Exception as e:
                print(f"Failed to send email to {recipient}: {e}")
        
        # Close the server connection
        server.quit()

    # Example usage:
    if __name__ == "__main__":
        sender_email = input("Sender email: ")
        app_password = input("App password: ")  # Use the app password here
        recipients = ['aditya27021999singh@gmail.com', 'aditya27021999singh@gmail.com', 'aditya27021999singh@gmail.com']  # Add the recipient emails here
        subject = input("Subject: ")
        message_body = input("Message: ")

        send_bulk_email(sender_email, app_password, recipients, subject, message_body)
        

def topsearch():
    from flask import Flask, request, render_template, jsonify
    import requests
    from bs4 import BeautifulSoup

    app = Flask(__name__)

    # Function to perform Google Search
    def google_search(query):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        params = {'q': query, 'num': 10}
        response = requests.get('https://www.google.com/search', headers=headers, params=params)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []

            for item in soup.find_all('div', attrs={'class': 'g'}, limit=10):
                title_element = item.find('h3')
                link_element = item.find('a')
                snippet_element = item.find('div', attrs={'class': 'IsZvec'})

                if title_element and link_element:
                    title = title_element.text
                    link = link_element['href']
                    snippet = snippet_element.text if snippet_element else 'No snippet'
                    results.append({
                        'title': title,
                        'link': link,
                        'snippet': snippet
                    })

                if len(results) >= 5:
                    break

            return results
        else:
            return None

    # Route for the homepage
    @app.route('/')
    def home():
        return render_template('pythontask.html')

    # Route for handling the search
    @app.route('/search', methods=['GET'])
    def search():
        query = request.args.get('query')
        if query:
            results = google_search(query)
            return jsonify(results)
        return jsonify({"error": "No query provided"})

    if __name__ == "__main__":
        app.run(debug=True)
        
def picClick():
    from flask import Flask, send_from_directory

    app = Flask(__name__)

    @app.route('/')
    def index():
        return send_from_directory('.', 'JSindex.html')

    if __name__ == '__main__':
        app.run(debug=True)
        
def searchEngine():
    from flask import Flask, send_from_directory

    app = Flask(__name__)

    @app.route('/')
    def index():
        return send_from_directory('.', 'JSindex.html')

    if __name__ == '__main__':
        app.run(debug=True)
        
def accessMic():
    from flask import Flask, send_from_directory

    app = Flask(__name__)

    @app.route('/')
    def index():
        return send_from_directory('.', 'JSindex.html')

    if __name__ == '__main__':
        app.run(debug=True)
        
def liveStream():
    from flask import Flask, send_from_directory

    app = Flask(__name__)

    @app.route('/')
    def index():
        return send_from_directory('.', 'JSindex.html')

    if __name__ == '__main__':
        app.run(debug=True)
        
def audiototext():
    from flask import Flask, send_from_directory

    app = Flask(__name__)

    @app.route('/')
    def index():
        return send_from_directory('.', 'JSindex.html')

    if __name__ == '__main__':
        app.run(debug=True)






    
    
    

        
       
    


        

if ch=='1':
    whatsapp()
elif ch=='2':
    text()
elif ch=='3':
    emails()
elif ch=='4':
    table()
elif ch=='5':
    bulkwhats()
elif ch=='6':
    speech()
elif ch=='7':
    images()
elif ch=='8':
    rainbow()
elif ch=='9':
    htmlcode()
elif ch=='10':
    search()
elif ch=='11':
    filter()
elif ch=='12':
    sunglass()
elif ch=='13':
    cornercrop()
elif ch=='14':
    numpyimg()
elif ch=='15':
    engtofrench()
elif ch=='16':
    faceblur()
elif ch=='17':
    VolControl()
elif ch=='18':
    location()
elif ch=='19':
    bulkEmail()
elif ch=='20':
    topsearch()
elif ch=='21':
    picClick()
elif ch=='22':
    searchEngine()
elif ch=='23':
    accessMic()
elif ch=='24':
    liveStream()
elif ch=='25':
    audiototext()
else:
    speech()
        
        

    





















     