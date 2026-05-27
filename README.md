This project is use to check the similarity between the faces between two images. Fours functions are included in this project- decode_images: it will decode the iamges coming from the frontend and make it available for opencv to proceed.
- detect_and_align: it will detect the face and if the face is not in the center it will re align it . check blur score and finally embedding. on the basis of embedding it will calculate similarity score and comparew the faces.
# To clone this project

Steps to clone : 
1. create python virtual environment: python -m venv task
2. activate virtual environment: task\scripts\activate
3. download libraries used in this project: pip install -r requirements.txt
4. run: python main.py
   
